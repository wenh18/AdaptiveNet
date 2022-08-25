"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
from copy import deepcopy

import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import copy
import numpy as np
from ._base import EncoderMixin


class ResNetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def get_block_info(self, block):
        # block = self.get_block(layer_idx, block_idx)
        return block.conv1.in_channels, block.conv3.in_channels, block.conv3.out_channels, block.conv2.stride[0]

    def get_multi_resnet_backbone(self):
        self.layer1, self.layerchoices1 = self.get_multi_resnet_layer(self.layer1)
        self.layer2, self.layerchoices2 = self.get_multi_resnet_layer(self.layer2)
        self.layer3, self.layerchoices3 = self.get_multi_resnet_layer(self.layer3)
        self.layer4, self.layerchoices4 = self.get_multi_resnet_layer(self.layer4)

    def update_multi_resnet_backbone(self):
        self.layer1, self.layerchoices1 = self.update_multi_resnet_layer(self.layer1)
        self.layer2, self.layerchoices2 = self.update_multi_resnet_layer(self.layer2)
        self.layer3, self.layerchoices3 = self.update_multi_resnet_layer(self.layer3)
        self.layer4, self.layerchoices4 = self.update_multi_resnet_layer(self.layer4)

    def update_multi_resnet_layer(self, layer):
        multiblocks = nn.ModuleList()
        block_choices = []
        # multi_block_idx = 0
        for block_idx in range(len(layer)):
            multiblocks.append(nn.ModuleList())
            multiblocks[-1].append(layer[block_idx][0])
            block_choice = 0
            this_in, this_mid, this_out, this_stride = self.get_block_info(layer[block_idx][0])
            if block_idx <= len(layer) - 2:
                next_block = layer[block_idx + 1][0]
                distill_next = copy.deepcopy(layer[block_idx][0])

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
                next_next_block = layer[block_idx + 2][0]
                distill_next_next = copy.deepcopy(layer[block_idx][0])

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
            return [self.generate_random_resnet_layer(self.layerchoices1, True), self.generate_random_resnet_layer(self.layerchoices2, True), self.generate_random_resnet_layer(self.layerchoices3, True),
                    self.generate_random_resnet_layer(self.layerchoices4, True)]
        return [self.generate_random_resnet_layer(self.layerchoices1), self.generate_random_resnet_layer(self.layerchoices2), self.generate_random_resnet_layer(self.layerchoices3),
                self.generate_random_resnet_layer(self.layerchoices4)]
    def generate_main_subnet(self):
        return [[0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices1)))], [0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices2)))],
                [0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices3)))],
                [0 for _ in range(len(self.generate_random_resnet_layer(self.layerchoices4)))]]

    def forward_resnet_layer(self, layer, x, layerchoice):
        blockidx = 0
        while blockidx < len(layer):
            x = layer[blockidx][layerchoice[blockidx]](x)
            if layerchoice[blockidx] == 1:
                blockidx += 2
            elif layerchoice[blockidx] == 2:
                blockidx += 3
            else:
                blockidx += 1
        return x

    def forward_distill(self, layer, x, layerchoice, x_teacher):
        blockidx = 0
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
        return x, x_teacher, num_distill_layers, distill_loss  # , features, feature_idxs
    def forward(self, x, subnet=None, distill=False):
        stages = self.get_stages()
        if not distill:
        # import pdb;pdb.set_trace()

            features = []
            x = stages[0](x)
            features.append(x)
            x = stages[1](x)
            features.append(x)
            x = self.maxpool(x)
            x = self.forward_resnet_layer(layer=self.layer1, x=x, layerchoice=subnet[0])
            features.append(x)
            x = self.forward_resnet_layer(layer=self.layer2, x=x, layerchoice=subnet[1])
            features.append(x)
            x = self.forward_resnet_layer(layer=self.layer3, x=x, layerchoice=subnet[2])
            features.append(x)
            x = self.forward_resnet_layer(layer=self.layer4, x=x, layerchoice=subnet[3])
            features.append(x)
            return features
        else:
            # import pdb;pdb.set_trace()
            # features = []
            # x = stages[0](x)
            # features.append(x)
            x_teacher = copy.deepcopy(x)
            x = stages[1](x)
            # features.append(x)
            x = self.maxpool(x)
            # x_teacher = copy.deepcopy(x)
            x_teacher = stages[1](x_teacher)
            x_teacher = self.maxpool(x_teacher)
            x1, x_teacher_1, num_distill_layers_1, distill_loss_1 = self.forward_distill(self.layer1, x,
                                                                                              subnet[0], x_teacher)

            x2, x_teacher_2, num_distill_layers_2, distill_loss_2 = self.forward_distill(self.layer2, x1,
                                                                                              subnet[1], x_teacher_1)
            x3, x_teacher_3, num_distill_layers_3, distill_loss_3 = self.forward_distill(self.layer3, x2,
                                                                                              subnet[2], x_teacher_2)
            x4, x_teacher_4, num_distill_layers_4, distill_loss_4 = self.forward_distill(self.layer4, x3,
                                                                                              subnet[3], x_teacher_3)
            # features += [x1, x2, x3, x4]
            # print(x2.shape, x3.shape, x4.shape)
            distill_loss = distill_loss_2 + distill_loss_3 + distill_loss_4
            distill_num = num_distill_layers_2 + num_distill_layers_3 + num_distill_layers_4
            if distill_num > 0.1:
                distill_loss /= distill_num
            # x_teacher =
            return distill_loss
    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


new_settings = {
    "resnet18": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet18-d92f0530.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth",  # noqa
    },
    "resnet50": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet50-16a12f1b.pth",  # noqa
    },
    "resnext50_32x4d": {
        "imagenet": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext50_32x4-72679e44.pth",  # noqa
    },
    "resnext101_32x4d": {
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x4-dc43570a.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth",  # noqa
    },
    "resnext101_32x8d": {
        "imagenet": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x8-2cfe2f8b.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x8-b4712904.pth",  # noqa
    },
    "resnext101_32x16d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
        "ssl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext101_32x16-15fffa57.pth",  # noqa
        "swsl": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth",  # noqa
    },
    "resnext101_32x32d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
    },
    "resnext101_32x48d": {
        "instagram": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
    },
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet18"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet34"],
        "params": {
            "out_channels": (3, 64, 64, 128, 256, 512),
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x8d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x16d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x32d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": pretrained_settings["resnext101_32x48d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
