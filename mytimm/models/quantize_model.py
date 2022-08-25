
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, prepare_qat, convert
from mytimm.models.MOE_modules import *
from mytimm.models.layers import create_attn


def create_aa(aa_layer, channels, stride=2, enable=True):
    if not aa_layer or not enable:
        return None
    return aa_layer(stride) if issubclass(aa_layer, nn.AvgPool2d) else aa_layer(channels=channels, stride=stride)


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class QuantBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(QuantBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.skip_add = nn.quantized.FloatFunctional()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    # def fuse_model(self):
    #     fuse_list = [[self.conv1, self.bn1, self.act1], [self.conv2, self.bn2, self.act2], [self.conv3, self.bn3, self.act3]]
    #     fuse_modules(fuse_list[0], inplace=True)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        # x += shortcut
        x = self.skip_add.add(x, shortcut)
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)
    for key, value in reassign.items():
        module._modules[key] = value


def _change_layer(layer, teachermodel=False):
    for blockidx in range(len(layer)):
        if teachermodel:
            temp_bottleneck = QuantBottleneck(inplanes=3, planes=64)
            temp_bottleneck.conv1 = layer[blockidx].conv1
            temp_bottleneck.conv2 = layer[blockidx].conv2
            temp_bottleneck.conv3 = layer[blockidx].conv3
            temp_bottleneck.bn1 = layer[blockidx].bn1
            temp_bottleneck.bn2 = layer[blockidx].bn2
            temp_bottleneck.bn3 = layer[blockidx].bn3
            temp_bottleneck.act1 = layer[blockidx].act1
            temp_bottleneck.act2 = layer[blockidx].act2
            temp_bottleneck.act3 = layer[blockidx].act3
            temp_bottleneck.downsample = layer[blockidx].downsample
            layer[blockidx] = temp_bottleneck

            # fuse conv+bn+relu
            fuse_modules(layer[blockidx], [['conv1', 'bn1', 'act1'], ['conv2', 'bn2', 'act2'], ['conv3', 'bn3', 'act3']], inplace=True)
            if layer[blockidx].downsample is not None:
                fuse_modules(layer[blockidx].downsample, ['0', '1'], inplace=True)

        else:
            for choiceidx in range(len(layer[blockidx])):
                temp_bottleneck = QuantBottleneck(inplanes=3, planes=64)
                temp_bottleneck.conv1 = layer[blockidx][choiceidx].conv1
                temp_bottleneck.conv2 = layer[blockidx][choiceidx].conv2
                temp_bottleneck.conv3 = layer[blockidx][choiceidx].conv3
                temp_bottleneck.bn1 = layer[blockidx][choiceidx].bn1
                temp_bottleneck.bn2 = layer[blockidx][choiceidx].bn2
                temp_bottleneck.bn3 = layer[blockidx][choiceidx].bn3
                temp_bottleneck.act1 = layer[blockidx][choiceidx].act1
                temp_bottleneck.act2 = layer[blockidx][choiceidx].act2
                temp_bottleneck.act3 = layer[blockidx][choiceidx].act3
                temp_bottleneck.downsample = layer[blockidx][choiceidx].downsample
                layer[blockidx][choiceidx] = temp_bottleneck

                # fuse conv+bn+relu
                fuse_modules(layer[blockidx][choiceidx], [['conv1', 'bn1', 'act1'], ['conv2', 'bn2', 'act2'], ['conv3', 'bn3', 'act3']], inplace=True)
                if layer[blockidx][choiceidx].downsample is not None:
                    fuse_modules(layer[blockidx][choiceidx].downsample, ['0', '1'], inplace=True)
    return layer


def quantize_resnet(model, static_post_training=True, teachermodel=False):
    model.act1 = nn.ReLU(inplace=False)
    # import pdb;pdb.set_trace()
    model.eval()
    fuse_modules(model, ['conv1', 'bn1', 'act1'])
    if not teachermodel:
        model.multiblocks = _change_layer(model.multiblocks)
    else:
        model.layer1 = _change_layer(model.layer1, True)
        model.layer2 = _change_layer(model.layer2, True)
        model.layer3 = _change_layer(model.layer3, True)
        model.layer4 = _change_layer(model.layer4, True)
    _replace_relu(model)
    model.eval()
    BACKEND = "fbgemm"
    model.qconfig = torch.quantization.get_default_qconfig(BACKEND)
    # model.qconfig = torch.quantization.default_qconfig
    if static_post_training:
        model = torch.quantization.prepare(model)
    else:
        prepare_qat(model, inplace=True)
    # import pdb;pdb.set_trace()
    return model


if __name__ == '__main__':
    import timm
    import torchvision.transforms as transforms
    import torchvision
    import numpy as np
    from torch.utils.data import Subset
    model = timm.create_model('resnet50', pretrained=True).cuda()
    model = quantize_resnet(model, teachermodel=True)
    for k,v in model.state_dict().items():
        print(k)
    model.load_state_dict(torch.load("../../output/quant/teacherresnet50epoch0batch3000.pth"), strict=True)
    # print(model)
    # x = torch.randn(64, 3, 224, 224).cuda()
    data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset_eval = torchvision.datasets.ImageFolder(root="../../../../project1/MultiBranchNet/mobilenet-yolov4-pytorch/"
                                                         "pytorch-image-models-master/data/imagenet/val",
                                                    transform=data_transform)
    idxs = np.random.choice(50000, 2000, replace=False).tolist()
    # print(idxs.shape)
    eval_set = Subset(dataset_eval, idxs)
    loader_eval = torch.utils.data.DataLoader(eval_set, 2, shuffle=True, num_workers=4)
    for b, (x, _ )in enumerate(loader_eval):
        x = x.cuda()
        x = QuantStub()(x)
        y = model(x)
        y = DeQuantStub()(y)
        print(y.shape)