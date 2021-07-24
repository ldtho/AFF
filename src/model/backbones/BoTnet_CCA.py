import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class Swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, silu=True,
                 bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.activation = nn.SiLU(inplace=True) if silu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], silu=False):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.SiLU(inplace=True) if silu else nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     output = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, silu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CoordAtt(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, silu=False):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(gate_channels // reduction_ratio)
        self.activation = nn.SiLU(inplace=True) if silu else nn.ReLU()

        self.conv_h = nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n_batch, C, H, W = x.size()
        # print('x.shape',x.shape)
        x_h = self.pool_h(x)
        # print('x_h.shape',x_h.shape)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # print('x_w.shape',x_w.shape)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.activation(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        att_h = self.conv_h(x_h).sigmoid()
        att_w = self.conv_w(x_w).sigmoid()

        out = identity * att_w * att_h
        return out


class CCA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types=['avg', 'max'], channel_att = False,spatial_att=False, coord_att=False,
                 silu=False):
        super(CCA, self).__init__()
        self.spatial_att = spatial_att
        self.coord_att = coord_att
        self.channel_att = channel_att
        if channel_att:
            self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types, silu=silu)
        if spatial_att:
            self.SpatialGate = SpatialGate()
        if coord_att:
            self.CoordAtt = CoordAtt(gate_channels, reduction_ratio, silu)

    def forward(self, x):
        x_out = x
        if self.channel_att:
            x_out = self.ChannelGate(x_out)
        if self.spatial_att:
            x_out = self.SpatialGate(x_out)
        if self.coord_att:
            x_out = self.CoordAtt(x_out)
        return x_out


if __name__ == '__main__':
    x = torch.rand(1, 32, 64, 64)
    model = CCA(32, reduction_ratio=8, pool_types=['avg', 'max'], channel_att = False,spatial_att=False, coord_att=True, silu=True)
    print(model)
    print(model(x).shape)
