import torch
import torch.nn as nn
import numpy as np

NORM_PPF_0_75 = 0.6745


class WeightQuantizer(nn.Module):

    def __init__(self, nbit, num_filters, method='QEM'):
        super().__init__()
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.num_filters = num_filters
        self.method = method
        init_basis = []
        n = num_filters * 3 * 3 if num_filters > 1 else 2
        base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
        for i in range(num_filters):
            t = [(2 ** j) * base for j in range(nbit)]
            init_basis.append(t)
        if method == 'QEM':
            self.basis = nn.Parameter(torch.Tensor(init_basis), requires_grad=False)
        else:
            self.basis = nn.Parameter(torch.Tensor(init_basis), requires_grad=True)
        num_levels = 2 ** nbit
        init_level_multiplier = []
        for i in range(num_levels):
            level_multiplier_i = [0. for j in range(nbit)]
            level_number = i
            for j in range(nbit):
                binary_code = level_number % 2
                if binary_code == 0:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.level_multiplier = nn.Parameter(torch.Tensor(init_level_multiplier), requires_grad=False)
        init_thrs_multiplier = []
        for i in range(1, num_levels):
            thrs_multiplier_i = [0. for j in range(num_levels)]
            thrs_multiplier_i[i - 1] = 0.5
            thrs_multiplier_i[i] = 0.5
            init_thrs_multiplier.append(thrs_multiplier_i)
        self.thrs_multiplier = nn.Parameter(torch.Tensor(init_thrs_multiplier), requires_grad=False)
        self.level_codes_channelwise = nn.Parameter(torch.zeros(num_filters, num_levels, nbit), requires_grad=False)
        self.eps = nn.Parameter(torch.eye(nbit), requires_grad=False)
        self.record = []

    def forward(self, x, training=False):
        if self.nbit == 0:
            return x
        nbit = self.nbit
        num_filters = self.num_filters
        num_levels = 2 ** self.nbit

        assert x.size(0) == num_filters

        levels = torch.mm(self.basis, self.level_multiplier.t())
        levels, sort_id = torch.topk(levels, k=num_levels, dim=1, largest=False)
        thrs = torch.mm(levels, self.thrs_multiplier.t())

        reshape_x = x.view(num_filters, -1)

        level_codes_channelwise = self.level_codes_channelwise
        for i in range(num_levels):
            eq = (sort_id == i).unsqueeze(2).expand(num_filters, num_levels, nbit)
            level_codes_channelwise = torch.where(eq,
                                                  self.level_multiplier[i].view(-1).expand_as(level_codes_channelwise),
                                                  level_codes_channelwise)
        y = torch.zeros_like(reshape_x) + levels[:, 0].view(-1, 1)
        bits_y = reshape_x.clone().unsqueeze(2).expand(num_filters, reshape_x.size(1), nbit)
        bits_y = bits_y * 0 - 1
        for i in range(num_levels - 1):
            gt = reshape_x >= thrs[:, i].view(-1, 1)
            y = torch.where(gt, levels[:, i + 1].view(-1, 1).expand_as(y), y)
            tt = gt.unsqueeze(2).expand(list(reshape_x.size()) + [nbit])
            bits_y = torch.where(tt, level_codes_channelwise[:, i + 1].view(num_filters, 1, nbit).expand_as(bits_y),
                                 bits_y)
        if training and self.method == 'QEM':
            BT = bits_y.view(num_filters, -1, nbit)
            B = BT.transpose(1, 2)
            BxBT = torch.bmm(B, BT)
            try:
                BxBT_inv = torch.inverse(BxBT)
            except RuntimeError:
                BxBT += self.eps
                BxBT_inv = torch.inverse(BxBT)
            else:
                BxX = torch.bmm(B, x.view(num_filters, -1, 1))
                new_basis = torch.bmm(BxBT_inv, BxX)
                new_basis = torch.topk(new_basis, k=nbit, dim=1, largest=False)[0]
                self.record.append(new_basis.view(num_filters, nbit).unsqueeze(0))
        y = y.view_as(x)
        if num_filters > 1:
            return y + x + x.detach() * -1
        else:
            t = torch.clamp(x, levels.min().item(), levels.max().item())
            return y + t + t.detach() * -1


class ActivationQuantizer(nn.Module):

    def __init__(self, nbit, method='QEM'):
        super().__init__()
        self.nbit = nbit
        if self.nbit == 0:
            return
        self.weight_quantizer = WeightQuantizer(nbit, num_filters=1, method=method)

    def forward(self, x, training=False):
        if self.nbit == 0:
            return x
        t = x.view(1, -1)
        y = self.weight_quantizer(t, training)
        y = y.view_as(x)
        return y


class QuantConv2d(nn.Conv2d):

    def __init__(self, w_bit=0, a_bit=0, method='QEM', **kwargs):
        super().__init__(**kwargs)
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.weight_quantizer = WeightQuantizer(w_bit, self.out_channels, method=method)
        self.activation_quantizer = ActivationQuantizer(a_bit, method=method)

    def forward(self, x):
        if (self.in_channels > 3):
            x = self.activation_quantizer(x, training=self.training)
            new_weight = self.weight_quantizer(self.weight, training=self.training)
        else:
            new_weight = self.weight
        y = nn.functional.conv2d(x, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y


if __name__ == '__main__':
    torch.manual_seed(0)
    l = QuantConv2d(w_bit=3, a_bit=0, method='BP', in_channels=4, out_channels=3, kernel_size=1)
    x = torch.randn(1, 4, 3, 3)
    y = l(x)
    loss = y.sum()
    loss.backward()
    print(l.weight_quantizer.basis.grad)
    print(l.weight.grad)