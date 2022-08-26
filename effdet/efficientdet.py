""" PyTorch EfficientDet model

Based on official Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
Paper: https://arxiv.org/abs/1911.09070

Hacked together by Ross Wightman
"""
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import List, Callable, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import create_conv2d, create_pool2d, get_act_layer
import time
from .anchors import get_feat_sizes
from .config import get_fpn_config, set_config_writeable, set_config_readonly
import copy
import numpy as np
_DEBUG = False
_USE_SCALE = False
_ACT_LAYER = get_act_layer('silu')

import torch_pruning as tp
class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding='', bias=False,
                 channel_multiplier=1.0, pw_kernel_size=1, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels, int(in_channels * channel_multiplier), kernel_size,
            stride=stride, dilation=dilation, padding=padding, depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier), out_channels, pw_kernel_size, padding=padding, bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False)


class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self, in_channels, out_channels, input_size, output_size, pad_type='',
            downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_bn=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module('conv', ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias, act_layer=None))

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type)
            else:
                if _USE_SCALE:  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module('downsample', down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
                else:
                    self.add_module('upsample', Interpolate2d(size=output_size, mode=upsample))


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_channels, inputs_offsets, output_size, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]['num_chs'], fpn_channels,
                input_size=feature_info[offset]['size'], output_size=output_size, pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, feat_sizes, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER,
                 apply_resample_bn=False, pre_act=True, separable_conv=True, redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc['feat_level']]) for fc in fpn_config.nodes]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                output_size=feat_sizes[fnode_cfg['feat_level']], pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=False, norm_layer=norm_layer, act_layer=act_layer)
            if pre_act:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x


class HeadNet(nn.Module):

    def __init__(self, config, num_outputs):
        super(HeadNet, self).__init__()
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, 'head_bn_level_first', False)
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_type = config.head_act_type if getattr(config, 'head_act_type', None) else config.act_type
        act_layer = get_act_layer(act_type) or _ACT_LAYER

        # Build convolution repeats
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        conv_kwargs = dict(
            in_channels=config.fpn_channels, out_channels=config.fpn_channels, kernel_size=3,
            padding=config.pad_type, bias=config.redundant_bias, act_layer=None, norm_layer=None)
        self.conv_rep = nn.ModuleList([conv_fn(**conv_kwargs) for _ in range(config.box_class_repeats)])

        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        # This can be organized with repeats first or feature levels first in module lists, the original models
        # and weights were setup with repeats first, levels first is required for efficient torchscript usage.
        self.bn_rep = nn.ModuleList()
        if self.bn_level_first:
            for _ in range(self.num_levels):
                self.bn_rep.append(nn.ModuleList([
                    norm_layer(config.fpn_channels) for _ in range(config.box_class_repeats)]))
        else:
            for _ in range(config.box_class_repeats):
                self.bn_rep.append(nn.ModuleList([
                    nn.Sequential(OrderedDict([('bn', norm_layer(config.fpn_channels))]))
                    for _ in range(self.num_levels)]))

        self.act = act_layer(inplace=True)

        # Prediction (output) layer. Has bias with special init reqs, see init fn.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(
            in_channels=config.fpn_channels, out_channels=num_outputs * num_anchors, kernel_size=3,
            padding=config.pad_type, bias=True, norm_layer=None, act_layer=None)
        self.predict = conv_fn(**predict_kwargs)

    @torch.jit.ignore()
    def toggle_bn_level_first(self):
        """ Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        with torch.no_grad():
            new_bn_rep = nn.ModuleList()
            for i in range(len(self.bn_rep[0])):
                bn_first = nn.ModuleList()
                for r in self.bn_rep.children():
                    m = r[i]
                    # NOTE original rep first model def has extra Sequential container with 'bn', this was
                    # flattened in the level first definition.
                    bn_first.append(m[0] if isinstance(m, nn.Sequential) else nn.Sequential(OrderedDict([('bn', m)])))
                new_bn_rep.append(bn_first)
            self.bn_level_first = not self.bn_level_first
            self.bn_rep = new_bn_rep

    @torch.jit.ignore()
    def _forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)  # this is not allowed in torchscript
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def _forward_level_first(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(self.bn_rep):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x)


def _init_weight(m, n='', ):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(m, n='', ):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(backbone.feature_info())]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
    return feature_info

def test_lat(block, input, test_times):
    # import pdb;pdb.set_trace()
    lats = []
    for i in range(test_times):
        t1 = time.time()
        y = block(input)
        torch.cuda.synchronize()
        t2 = time.time() - t1
        if i > 100:
            lats.append(t2)
        del y
    # del x
    return np.mean(lats)
def test_resnet_layer(layer, x):
    lats = []
    for blockidx in range(len(layer)):
        lat_choices = []
        for choiceidx in range(len(layer[blockidx])):
            lat_choices.append(test_lat(layer[blockidx][choiceidx], x, 200))
        x = layer[blockidx][0](x)
        lats.append(lat_choices)
    return lats, x

class EfficientDet(nn.Module):

    def __init__(self, config, pretrained_backbone=True, alternate_init=False):
        super(EfficientDet, self).__init__()
        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name, features_only=True,
            out_indices=self.config.backbone_indices or (2, 3, 4),
            pretrained=pretrained_backbone, **config.backbone_args)
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(self.config, feature_info)
        self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
        self.box_net = HeadNet(self.config, num_outputs=4)


        self.get_multi_resnet_backbone()
        # self.reshape_model_to_subnet([[2, 99, 99, 2, 99, 99, 0], [2, 99, 99, 2, 99, 99], [2, 99, 99]])

        # TODO:change self.backbone to elastic supermodel, so that we can load imagenet pretrained model
        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    @torch.jit.ignore()
    def reset_head(self, num_classes=None, aspect_ratios=None, num_scales=None, alternate_init=False):
        reset_class_head = False
        reset_box_head = False
        set_config_writeable(self.config)
        if num_classes is not None:
            reset_class_head = True
            self.config.num_classes = num_classes
        if aspect_ratios is not None:
            reset_box_head = True
            self.config.aspect_ratios = aspect_ratios
        if num_scales is not None:
            reset_box_head = True
            self.config.num_scales = num_scales
        set_config_readonly(self.config)

        if reset_class_head:
            self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
            for n, m in self.class_net.named_modules(prefix='class_net'):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

        if reset_box_head:
            self.box_net = HeadNet(self.config, num_outputs=4)
            for n, m in self.box_net.named_modules(prefix='box_net'):
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def get_multi_resnet_backbone(self):
        layer2 = []
        for layer in self.backbone.layer1:
            layer2.append(layer)
        for layer in self.backbone.layer2:
            layer2.append(layer)
        self.backbone.layer2 = nn.Sequential(*layer2)
        del self.backbone.layer1
        self.backbone.layer2, self.layerchoices2 = self.get_multi_resnet_layer(self.backbone.layer2)
        self.backbone.layer3, self.layerchoices3 = self.get_multi_resnet_layer(self.backbone.layer3)
        self.backbone.layer4, self.layerchoices4 = self.get_multi_resnet_layer(self.backbone.layer4)

    def get_block_info(self, block):
        # block = self.get_block(layer_idx, block_idx)
        return block.conv1.in_channels, block.conv3.in_channels, block.conv3.out_channels, block.conv2.stride[0]

    def get_pruned_layers(self, pruned_layers):
        for ratio_idx in range(len(pruned_layers)):
            for blockidx in range(len(pruned_layers[ratio_idx][0])):
                self.backbone.layer2[blockidx].append(pruned_layers[ratio_idx][0][blockidx])
            for blockidx in range(len(pruned_layers[ratio_idx][1])):
                self.backbone.layer3[blockidx].append(pruned_layers[ratio_idx][1][blockidx])
            for blockidx in range(len(pruned_layers[ratio_idx][2])):
                self.backbone.layer4[blockidx].append(pruned_layers[ratio_idx][2][blockidx])


    def get_multi_resnet_layer(self, layer):
        multiblocks = nn.ModuleList()
        block_choices = []
        # multi_block_idx = 0
        for block_idx in range(len(layer)):
            multiblocks.append(nn.ModuleList())
            multiblocks[-1].append(layer[block_idx])
            block_choice = 0
            this_in, this_mid, this_out, this_stride = self.get_block_info(layer[block_idx])
            if block_idx <= len(layer) - 2:
                next_block = layer[block_idx + 1]
                distill_next = copy.deepcopy(layer[block_idx])

                next_in, next_mid, next_out, next_stride = self.get_block_info(next_block)
                stride = max(this_stride, next_stride)
                if stride != this_stride:
                    distill_next.conv2 = nn.Conv2d(this_mid, this_mid, kernel_size=(3, 3), stride=(stride, stride),
                                                   padding=(1, 1), bias=False)
                if this_out != next_out:
                    distill_next.conv3 = nn.Conv2d(this_mid, next_out, kernel_size=(1, 1), bias=False)
                    distill_next.bn3 = nn.BatchNorm2d(next_out)
                if this_in != next_out:
                    distill_next.downsample = nn.Sequential(*[
                        nn.Conv2d(
                            this_in, next_out, (1, 1), stride=stride,
                            bias=False),
                        nn.BatchNorm2d(next_out)
                    ])
                else:
                    distill_next.downsample = None
                block_choice += 1
                multiblocks[-1].append(distill_next)
                # self.block_choices[-1].append(1)

            if block_idx <= len(layer) - 3:
                next_next_block = layer[block_idx + 2]
                distill_next_next = copy.deepcopy(layer[block_idx])

                next_next_in, next_next_mid, next_next_out, next_next_stride = self.get_block_info(next_next_block)
                stride = max(this_stride, next_stride, next_next_stride)
                if stride != this_stride:
                    distill_next_next.conv2 = nn.Conv2d(this_mid, this_mid, kernel_size=(3, 3),
                                                        stride=(stride, stride), padding=(1, 1), bias=False)
                if this_out != next_next_out:
                    distill_next_next.conv3 = nn.Conv2d(this_mid, next_next_out, kernel_size=(1, 1), bias=False)
                    distill_next_next.bn3 = nn.BatchNorm2d(next_next_out)
                if this_in != next_next_out:
                    distill_next_next.downsample = nn.Sequential(*[
                        nn.Conv2d(
                            this_in, next_next_out, (1, 1), stride=stride,
                            bias=False),
                        nn.BatchNorm2d(next_next_out)
                    ])
                else:
                    distill_next_next.downsample = None
                block_choice += 1
                multiblocks[-1].append(distill_next_next)
                # self.block_choices[-1].append(2)

            block_choices.append(block_choice)
        return multiblocks, block_choices  # block_choices = [1, 0, 1, 2, ...]

    def generate_random_resnet_layer(self, layerchoices, min_lat=False):
        blockidx = 0
        sublayer = []
        while blockidx < len(layerchoices):
            if min_lat:
                choice = layerchoices[blockidx]
            else:
                choices = [i for i in range(layerchoices[blockidx] + 1)]
                # choices += [-1, -2]  # for pruned blocks
                choice = np.random.choice(choices)
            if choice == 1:
                sublayer += [1, 99]
                blockidx += 2
            elif choice == 2:
                sublayer += [2, 99, 99]
                blockidx += 3
            else:
                sublayer.append(choice)
                blockidx += 1
        return sublayer

    def generate_random_subnet(self, min_lat=False):
        if min_lat:
            return [self.generate_random_resnet_layer(self.layerchoices2, True), self.generate_random_resnet_layer(self.layerchoices3, True),
                    self.generate_random_resnet_layer(self.layerchoices4, True)]
        return [self.generate_random_resnet_layer(self.layerchoices2), self.generate_random_resnet_layer(self.layerchoices3),
                self.generate_random_resnet_layer(self.layerchoices4)]

    def generate_main_subnet(self):
        return [[0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices2)))],
                [0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices3)))],
                [0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices4)))]]

    def forward_resnet_layer(self, layer, x, layerchoice, x_teacher):
        blockidx = 0
        # features, feature_idxs = [], []
        flag_distill = True if x_teacher is not None else False
        # if flag_distill:
        num_distill_layers, distill_loss = 0, 0
        while blockidx < len(layer):
            x = layer[blockidx][layerchoice[blockidx]](x)
            if flag_distill:
                for skippidx in range(layerchoice[blockidx] + 1):
                    x_teacher = layer[blockidx + skippidx][0](x_teacher)
                if layerchoice[blockidx] != 0:
                    distill_loss += nn.MSELoss()(x_teacher, x)
                    num_distill_layers += 1
            if layerchoice[blockidx] == 1:
                blockidx += 2
                # feature_idxs.append(blockidx - 1)
                # features.append(x)
            elif layerchoice[blockidx] == 2:
                blockidx += 3
                # feature_idxs.append(blockidx - 1)
                # features.append(x)
            else:
                blockidx += 1
        return x, x_teacher, num_distill_layers, distill_loss#, features, feature_idxs

    def reshape_model_to_subnet(self, subnet):
        '''
        this method is only used when we want to measure the GPU memory we used
        '''
        layer2, layer3, layer4 = [], [], []
        for blockidx in range(len(subnet[0])):
            if subnet[0][blockidx] != 99:
                print(blockidx, ";;;")
                layer2.append(self.backbone.layer2[blockidx][subnet[0][blockidx]])
        for blockidx in range(len(subnet[1])):
            if subnet[1][blockidx] != 99:
                print(blockidx, ";;;")
                layer3.append(self.backbone.layer3[blockidx][subnet[1][blockidx]])
        for blockidx in range(len(subnet[2])):
            if subnet[2][blockidx] != 99:
                print(blockidx, ";;;")
                layer4.append(self.backbone.layer4[blockidx][subnet[2][blockidx]])
        self.backbone.layer2 = nn.Sequential(*layer2)
        self.backbone.layer3 = nn.Sequential(*layer3)
        self.backbone.layer4 = nn.Sequential(*layer4)

    @torch.jit.ignore()
    def toggle_head_bn_level_first(self):
        """ Toggle the head batchnorm layers between being access with feature_level first vs repeat
        """
        self.class_net.toggle_bn_level_first()
        self.box_net.toggle_bn_level_first()

    def forward_resnet_backbone(self, x, subnet, distill):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x_teacher = copy.deepcopy(x) if distill else None
        x2, x_teacher_2, num_distill_layers_2, distill_loss_2 = self.forward_resnet_layer(self.backbone.layer2, x, subnet[0], x_teacher)
        x3, x_teacher_3, num_distill_layers_3, distill_loss_3 = self.forward_resnet_layer(self.backbone.layer3, x2, subnet[1], x_teacher_2)
        x4, x_teacher_4, num_distill_layers_4, distill_loss_4 = self.forward_resnet_layer(self.backbone.layer4, x3, subnet[2], x_teacher_3)
        # print(x2.shape, x3.shape, x4.shape)
        if distill:
            distill_loss = distill_loss_2 + distill_loss_3 + distill_loss_4
            distill_num = num_distill_layers_2 + num_distill_layers_3 + num_distill_layers_4
            if distill_num > 0.1:
                distill_loss /= distill_num
            # x_teacher =
            return [x2, x3, x4], distill_loss, distill_num, [x_teacher_2, x_teacher_3, x_teacher_4]
        return [x2, x3, x4]

    def forward_reshaped(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x2 = self.backbone.layer2(x)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return [x2, x3, x4]


    def forward(self, x, resnet=True, subnet=None, distill=True, distill_head=True, reshaped=False):
        if resnet:
            if reshaped:  # testing the GPU memory the model used
                x = self.forward_reshaped(x)
            elif distill:  # distilling, namely the first stage
                x, distill_loss, distill_num, x_teacher = self.forward_resnet_backbone(x, subnet, distill)
                distill_loss *= 2  # 2:hyp that determines how much the backbone distillation loss matters
                # import pdb;pdb.set_trace()
            else:  # finetuning, namely the second stage
                x = self.forward_resnet_backbone(x, subnet, distill)
        # x = self.backbone(x)
        x = self.fpn(x)
        if distill and distill_head and not reshaped:
            x_teacher = self.fpn(x_teacher)
            fpn_loss = 0
            for i in range(len(x)):
                fpn_loss += (nn.MSELoss()(x_teacher[i], x[i])) / len(x)
            distill_loss += fpn_loss
        x_class = self.class_net(x)
        if distill and distill_head and not reshaped:
            teacher_class = self.class_net(x_teacher)
            cls_loss = 0
            for i in range(len(x_class)):
                cls_loss += (nn.MSELoss()(teacher_class[i], x_class[i])) / len(x_class)
            distill_loss += cls_loss
        x_box = self.box_net(x)
        if distill and distill_head and not reshaped:
            teacher_box = self.box_net(x_teacher)
            box_loss = 0
            for i in range(len(x_box)):
                box_loss += nn.MSELoss()(teacher_box[i], x_box[i]) / len(x_box)
            distill_loss += box_loss
        if distill and not reshaped:
            # if distill_head:
            #     print((distill_loss-fpn_loss-cls_loss-box_loss)/distill_loss, fpn_loss/distill_loss, cls_loss/distill_loss, box_loss/distill_loss)
            del x_teacher
            return x_class, x_box, distill_loss
        return x_class, x_box


# import torch
# m = torch.load()