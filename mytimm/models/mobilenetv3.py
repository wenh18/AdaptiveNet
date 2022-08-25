""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mytimm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .efficientnet_blocks import SqueezeExcite, InvertedResidual
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from .features import FeatureInfo, FeatureHooks
from .helpers import build_model_with_cfg, default_cfg_for_features
from .layers import SelectAdaptivePool2d, Linear, create_conv2d, get_act_fn, hard_sigmoid, make_divisible
from .registry import register_model
import copy
__all__ = ['MobileNetV3', 'MobileNetV3Features']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenetv3_large_075': _cfg(url=''),
    'mobilenetv3_large_100': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
    'mobilenetv3_large_100_miil': _cfg(
        interpolation='bilinear', mean=(0, 0, 0), std=(1, 1, 1),
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/mytimm/mobilenetv3_large_100_1k_miil_78_0.pth'),
    'mobilenetv3_large_100_miil_in21k': _cfg(
        interpolation='bilinear', mean=(0, 0, 0), std=(1, 1, 1),
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/mytimm/mobilenetv3_large_100_in21k_miil.pth', num_classes=11221),

    'mobilenetv3_small_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth',
        interpolation='bicubic'),

    'mobilenetv3_rw': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth',
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95),

    "lcnet_035": _cfg(),
    "lcnet_050": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth',
        interpolation='bicubic',
    ),
    "lcnet_075": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth',
        interpolation='bicubic',
    ),
    "lcnet_100": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth',
        interpolation='bicubic',
    ),
    "lcnet_150": _cfg(),
}


class MBV3Teacher(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(
            self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280,
            head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True,
            round_chs_fn=round_channels, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(MBV3Teacher, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, stage):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # if stage == 0:
            # x = self.blocks[](x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x, stage):
        x = self.forward_features(x, stage)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class MobileNetV3(nn.Module):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(
            self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280,
            head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True,
            round_chs_fn=round_channels, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        # self.train_stages = [6, 12, 15, 15]   # last 16 represents full length network
        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.cfgs = [
            #`   k,   t,   c, SE,HS,s
                # 208,208,16 -> 208,208,16
                [3,   1,  16, 0, 0, 1], # not changeable

                # 208,208,16 -> 104,104,24
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],

                # 104,104,24 -> 52,52,40
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],

                # 52,52,40 -> 26,26,80
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],

                # 26,26,80 -> 26,26,112
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],

                # 26,26,112 -> 13,13,160
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
        ]
        self.multiblocks = nn.ModuleList()
        block_idx = 0
        input_channel = 16
        self.block_state_dict = [[]]
        choices = 1
        for stage in self.blocks:
            # print("**************************************")
            for block in stage:
                temp = 1
                self.multiblocks.append(nn.ModuleList())
                self.multiblocks[-1].append(block)
                if block_idx != 0 and block_idx < len(self.cfgs):
                    state_dict = [0, 0, 0]
                    k, t, c, use_se, use_hs, s = self.cfgs[block_idx]
                    output_channel = make_divisible(c)
                    mid_chs = make_divisible(input_channel * t)
                    if output_channel == input_channel:
                        self.multiblocks[-1].append(nn.Identity())
                        state_dict[0] = 1
                        # temp += 1
                    if block_idx <= len(self.cfgs) - 2:
                        next_stride = self.cfgs[block_idx + 1][5]
                        next_output_channel = self.cfgs[block_idx+1][2]

                        stride = max(next_stride, s)
                        # if stride == 1:
                        distill_next = copy.deepcopy(block)
                        distill_next.conv_dw = create_conv2d(
                            mid_chs, mid_chs, k, stride=stride, dilation=block.dilation,
                            padding=block.pad_type, depthwise=True, **block.conv_kwargs)
                        # self.bn2 = norm_layer(mid_chs)
                        distill_next.has_residual = (input_channel == next_output_channel and stride == 1)
                        distill_next.conv_pwl = create_conv2d(mid_chs, next_output_channel, 1, padding=block.pad_type, **block.conv_kwargs)
                        distill_next.bn3 = norm_layer(next_output_channel)
                        self.multiblocks[-1].append(distill_next)
                        state_dict[1] = 1
                        temp += 1
                    if block_idx <= len(self.cfgs) - 3:
                        next_stride = self.cfgs[block_idx + 1][5]
                        next_next_stride = self.cfgs[block_idx + 2][5]
                        if s + next_stride + next_next_stride <= 4:
                            next_next_output_channel = self.cfgs[block_idx + 2][2]
                            stride = max(s, next_stride, next_next_stride)
                            distill_next_next = copy.deepcopy(block)
                            distill_next_next.conv_dw = create_conv2d(
                                mid_chs, mid_chs, k, stride=stride, dilation=block.dilation,
                                padding=block.pad_type, depthwise=True, **block.conv_kwargs)
                            distill_next_next.has_residual = (input_channel == next_next_output_channel and stride == 1)
                            distill_next_next.conv_pwl = create_conv2d(mid_chs, next_next_output_channel, 1, padding=block.pad_type, **block.conv_kwargs)
                            distill_next_next.bn3 = norm_layer(next_next_output_channel)
                            self.multiblocks[-1].append(distill_next_next)
                            state_dict[2] = 1
                        temp += 1
                    choices *= temp
                        # block = InvertedResidual(input_channel, next_output_channel, k, stride, exp_ratio=t, )
                    input_channel = output_channel
                    self.block_state_dict.append(state_dict)
                block_idx += 1
            # print("**************************************")
        # print("**************************************")
        # print(self.multiblocks)
        print("**************************************")
        print(choices)
        self.feature_info = builder.features
        head_chs = builder.in_chs
        del self.blocks
        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        print("----------------------this is mbv3-----------------------------")
        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def generate_random_subnet(self, stage=0):
        subnet = [0]
        subnet_choice = [0]  # 0->origin, 1->skip, 2->distill two layers, 3->distill three layers
        i = 1
        subnet_length = self.train_stages[stage]
        while i < subnet_length:
            choices = [0]
            if self.block_state_dict[i][0] == 1:
                choices.append(1)  # skip
            if self.block_state_dict[i][1] == 1:
                choices.append(2)  # distill 2
            if self.block_state_dict[i][2] == 1:
                choices.append(3)
            choice = np.random.choice(choices)
            subnet_choice.append(choice)
            if choice == 0 or choice == 1:
                subnet.append(choice)  # 当能使用原模块和跳过时，choice对应的就是modulelist中该模块的下标
                i += 1
            elif choice == 2:
                if self.block_state_dict[i][0] == 1:  # 当有跳过选项时
                    subnet.append(choice)
                else:
                    subnet.append(choice - 1)
                subnet.append(99)  # 无意义占位符
                subnet_choice.append(99)
                i += 2
            elif choice == 3:  # 能蒸三层时一定可以蒸两层
                if self.block_state_dict[i][0] == 1:  # 当有跳过选项时
                    subnet.append(choice)
                else:
                    subnet.append(choice - 1)
                subnet = subnet + [99, 99]
                subnet_choice = subnet_choice + [99, 99]
                i += 3
        if len(subnet) > subnet_length:
            while subnet[-1] == 99:
                subnet.pop()
                subnet_choice.pop()
            subnet.pop()
            subnet_choice.pop()
        if stage == 1 and len(subnet) == 10:
            subnet.append(0)
            subnet_choice.append(0)
        if stage >= 2:
            subnet.append(0)
            subnet_choice.append(0)

        while len(subnet) < (self.train_stages[3] + 1):
            subnet.append(0)
            subnet_choice.append(0)
            i += 1

        for i in range(len(subnet_choice)):
            if subnet_choice[i] == 1:
                subnet[i] = 0
                subnet_choice[i] = 0
        return subnet, subnet_choice

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, subnet, subnet_choice, train_stage=0):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # x = self.multiblocks[0](x)
        features = []
        feature_idx_outs = []
        block_cls = []
        i = 0
        while i < len(subnet):  # 16
            # print(i)
            x = self.multiblocks[i][subnet[i]](x)
            if subnet_choice[i] == 2:
                i += 2
                features.append(x)
                feature_idx_outs.append(i)
                block_cls.append(2)
            elif subnet_choice[i] == 3:
                i += 3
                features.append(x)
                feature_idx_outs.append(i)
                block_cls.append(3)
            else:
                i += 1
        if train_stage <= 2:
            return x
        # elif train_stage == 3:
        #     x = self.mul
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x, features, feature_idx_outs, block_cls

    def forward(self, x, subnet, subnet_choice, train_stage=0):
        x, features, feature_idx_outs, block_cls = self.forward_features(x, subnet, subnet_choice, train_stage)
        if train_stage < 3:
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return x
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x), features, feature_idx_outs, block_cls


class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor

    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(self, block_args, out_indices=(0, 1, 2, 3, 4), feature_location='bottleneck', in_chans=3,
                 stem_size=16, fix_stem=False, output_stride=32, pad_type='', round_chs_fn=round_channels,
                 se_from_exp=True, act_layer=None, norm_layer=None, se_layer=None, drop_rate=0., drop_path_rate=0.):
        super(MobileNetV3Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer,
            drop_path_rate=drop_path_rate, feature_location=feature_location)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_mnv3(variant, pretrained=False, teacher_model=False, **kwargs):
    # import pdb;pdb.set_trace()
    features_only = False
    if teacher_model:
        model_cls = MBV3Teacher
    else:
        model_cls = MobileNetV3
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        features_only = True
        kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'head_bias', 'global_pool')
        model_cls = MobileNetV3Features
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=not features_only,
        kwargs_filter=kwargs_filter,
        **kwargs)
    if features_only:
        model.default_cfg = default_cfg_for_features(model.default_cfg)
    return model


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid'),
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model

def _gen_teacher_mbv3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    num_features = 1280
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    # import pdb;pdb.set_trace()
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetv3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNetV3
    Paper: `FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining`
        - https://arxiv.org/abs/2006.02049
    FIXME untested, this is a preliminary impl of some FBNet-V3 variants.
    """
    vl = variant.split('_')[-1]
    if vl in ('a', 'b'):
        stem_size = 16
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k5_s2_e4_c24', 'ir_r3_k5_s1_e2_c24'],
            ['ir_r1_k5_s2_e5_c40_se0.25', 'ir_r4_k5_s1_e3_c40_se0.25'],
            ['ir_r1_k5_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c120_se0.25', 'ir_r5_k5_s1_e3_c120_se0.25'],
            ['ir_r1_k3_s2_e6_c184_se0.25', 'ir_r5_k5_s1_e4_c184_se0.25', 'ir_r1_k5_s1_e6_c224_se0.25'],
            ['cn_r1_k1_s1_c1344'],
        ]
    elif vl == 'd':
        stem_size = 24
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k3_s2_e5_c24', 'ir_r5_k3_s1_e2_c24'],
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r4_k3_s1_e3_c40_se0.25'],
            ['ir_r1_k3_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c128_se0.25', 'ir_r6_k5_s1_e3_c128_se0.25'],
            ['ir_r1_k3_s2_e6_c208_se0.25', 'ir_r5_k5_s1_e5_c208_se0.25', 'ir_r1_k5_s1_e6_c240_se0.25'],
            ['cn_r1_k1_s1_c1440'],
        ]
    elif vl == 'g':
        stem_size = 32
        arch_def = [
            ['ds_r3_k3_s1_e1_c24'],
            ['ir_r1_k5_s2_e4_c40', 'ir_r4_k5_s1_e2_c40'],
            ['ir_r1_k5_s2_e4_c56_se0.25', 'ir_r4_k5_s1_e3_c56_se0.25'],
            ['ir_r1_k5_s2_e5_c104', 'ir_r4_k3_s1_e3_c104'],
            ['ir_r1_k3_s1_e5_c160_se0.25', 'ir_r8_k5_s1_e3_c160_se0.25'],
            ['ir_r1_k3_s2_e6_c264_se0.25', 'ir_r6_k5_s1_e5_c264_se0.25', 'ir_r2_k5_s1_e6_c288_se0.25'],
            ['cn_r1_k1_s1_c1728'],
        ]
    else:
        raise NotImplemented
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.95)
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=round_chs_fn)
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1984,
        head_bias=False,
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        se_from_exp=False,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


@register_model
def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100_miil(pretrained=False, **kwargs):
    """ MobileNet V3
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_large_100_miil_in21k(pretrained=False, **kwargs):
    """ MobileNet V3, 21k pretraining
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil_in21k', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_050(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet V3 """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def tf_mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_b(pretrained=False, **kwargs):
    """ FBNetV3-B """
    model = _gen_fbnetv3('fbnetv3_b', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_d(pretrained=False, **kwargs):
    """ FBNetV3-D """
    model = _gen_fbnetv3('fbnetv3_d', pretrained=pretrained, **kwargs)
    return model


@register_model
def fbnetv3_g(pretrained=False, **kwargs):
    """ FBNetV3-G """
    model = _gen_fbnetv3('fbnetv3_g', pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_035(pretrained=False, **kwargs):
    """ PP-LCNet 0.35"""
    model = _gen_lcnet('lcnet_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_050(pretrained=False, **kwargs):
    """ PP-LCNet 0.5"""
    model = _gen_lcnet('lcnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_075(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_100(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def lcnet_150(pretrained=False, **kwargs):
    """ PP-LCNet 1.5"""
    model = _gen_lcnet('lcnet_150', 1.5, pretrained=pretrained, **kwargs)
    return model

# if __name__ == '__main__':
    # arch_def = [
    #     # stage 0, 112x112 in
    #     ['ds_r1_k3_s1_e1_c16_nre'],  # relu
    #     # stage 1, 112x112 in
    #     ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
    #     # stage 2, 56x56 in
    #     ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
    #     # stage 3, 28x28 in
    #     ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
    #     # stage 4, 14x14in
    #     ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
    #     # stage 5, 14x14in
    #     ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
    #     # stage 6, 7x7 in
    #     ['cn_r1_k1_s1_c960'],  # hard-swish
    # ]
    # block_args = decode_arch_def(arch_def)
    # print(block_args)