import functools
import torch
import torch.nn as nn
import math
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch import Tensor
# class MOEBottleneck(nn.Module):


# https://github.com/nibuiro/CondConv-pytorch/blob/master/condconv/condconv.py
class MOEconv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MOEconv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        # self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        # self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs, expert_weights):
        expert_weights = torch.tensor(expert_weights)
        # b, _, _, _ = inputs.size()
        # res = []
        # for input in inputs:
        #     input = input.unsqueeze(0)
        #     pooled_inputs = self._avg_pooling(input)
        #     routing_weights = self._routing_fn(pooled_inputs)
        kernels = torch.sum(expert_weights[:, None, None, None, None] * self.weight, 0)
        out = self._conv_forward(inputs, kernels)
            # res.append(out)
        return out  # torch.cat(res, dim=0)


class MOEGroupNormalization(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_experts: int, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MOEGroupNormalization, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.num_experts = num_experts
        if self.affine:
            self.weight = Parameter(torch.empty(num_experts, num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_experts, num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, expert_weights):
        expert_weights = torch.tensor(expert_weights)
        weight = torch.sum(expert_weights[:, None] * self.weight, 0)
        bias = torch.sum(expert_weights[:, None] * self.bias, 0)
        return F.group_norm(
            input, self.num_groups, weight, bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)
        

class MOEClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, num_expert: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MOEClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((num_expert, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(num_expert, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, expert_weights) -> Tensor:
        expert_weights = torch.tensor(expert_weights)
        weight = torch.sum(expert_weights[:, None, None] * self.weight, 0)
        if self.bias is not None:
            bias = torch.sum(expert_weights[:, None] * self.bias, 0)
            return F.linear(input, weight, bias)
        else:
            return F.linear(input, weight, None)


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# https://github.com/ldeecke/mn-torch/blob/master/nn/ops.py
class _ModeNormalization(nn.Module):
    def __init__(self, dim, num_experts, eps):
        super(_ModeNormalization, self).__init__()
        self.eps = eps
        self.dim = dim
        self.num_experts = num_experts

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        # self.phi = lambda x: x.mean(3).mean(2)


# https://github.com/ldeecke/mn-torch/blob/master/nn/ops.py
class MOEnorm(_ModeNormalization):
    """
    An implementation of mode normalization.
    Input samples x are allocated into individual modes (their number is controlled by n_components) by a gating network
     samples belonging together are jointly normalized and then passed back to the network.
    args:
        dim:                int
        momentum:           float
        n_components:       int
        eps:                float
    """

    def __init__(self, dim, num_experts, momentum=0.1, eps=1.e-5):
        super(MOEnorm, self).__init__(dim, num_experts, eps)

        self.momentum = momentum

        self.x_ra = torch.zeros(num_experts, 1, dim, 1, 1).cuda()
        self.x2_ra = torch.zeros(num_experts, 1, dim, 1, 1).cuda()

        self.W = torch.nn.Linear(dim, num_experts)
        self.W.weight.data = torch.ones(num_experts, dim) / num_experts + .01 * torch.randn(num_experts, dim)
        self.softmax = torch.nn.Softmax(dim=1)

        self.weighted_mean = lambda w, x, n: (w * x).mean(3, keepdim=True).mean(2, keepdim=True).sum(0,
                                                                                                     keepdim=True) / n

    def forward(self, x):
        g = self._g(x)
        n_k = torch.sum(g, dim=1).squeeze()

        if self.training:
            self._update_running_means(g.detach(), x.detach())

        x_split = torch.zeros(x.size()).cuda().to(x.device)

        for k in range(self.num_experts):
            if self.training:
                mu_k = self.weighted_mean(g[k], x, n_k[k])
                var_k = self.weighted_mean(g[k], (x - mu_k) ** 2, n_k[k])
            else:
                mu_k, var_k = self._mu_var(k)
                mu_k = mu_k.to(x.device)
                var_k = var_k.to(x.device)

            x_split += g[k] * ((x - mu_k) / torch.sqrt(var_k + self.eps))

        x = self.alpha * x_split + self.beta

        return x

    def _g(self, x):
        """
        Image inputs are first flattened along their height and width dimensions by phi(x), then mode memberships are determined via a linear transformation, followed by a softmax activation. The gates are returned with size (k, n, c, 1, 1).
        args:
            x:          torch.Tensor
        returns:
            g:          torch.Tensor
        """
        g = self.softmax(self.W(self.phi(x))).transpose(0, 1)[:, :, None, None, None]
        return g

    def _mu_var(self, k):
        """
        At test time, this function is used to compute the k'th mean and variance from weighted running averages of x and x^2.
        args:
            k:              int
        returns:
            mu, var:        torch.Tensor, torch.Tensor
        """
        mu = self.x_ra[k]
        var = self.x2_ra[k] - (self.x_ra[k] ** 2)
        return mu, var

    def _update_running_means(self, g, x):
        """
        Updates weighted running averages. These are kept and used to compute estimators at test time.
        args:
            g:              torch.Tensor
            x:              torch.Tensor
        """
        n_k = torch.sum(g, dim=1).squeeze()

        for k in range(self.num_experts):
            x_new = self.weighted_mean(g[k], x, n_k[k])
            x2_new = self.weighted_mean(g[k], x ** 2, n_k[k])

            # ensure that tensors are on the right devices
            self.x_ra = self.x_ra.to(x_new.device)
            self.x2_ra = self.x2_ra.to(x2_new.device)
            self.x_ra[k] = self.momentum * x_new + (1 - self.momentum) * self.x_ra[k]
            self.x2_ra[k] = self.momentum * x2_new + (1 - self.momentum) * self.x2_ra[k]

# class MOEConvBN(nn.Module):
#     def __init__(self):