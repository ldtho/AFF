import math

import torch
import torch.nn as nn
import torch.nn.functional as F

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3]
]

phi_values = {
    # (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5)
}


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = Swish()

    def forward(self, x):
        x = self.silu(self.bn(self.cnn(x)))
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduce_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # CxHxW -> Cx1x1
            nn.Conv2d(in_channels, reduce_dim, 1),
            Swish(),
            nn.Conv2d(reduce_dim, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, reduce_rate=4):
        super(SelfAttention, self).__init__()
        self.reduce_conv = nn.Conv2d(channels, channels // reduce_rate, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        print('x.shape', x.shape)
        q = self.reduce_conv(x)
        k = self.reduce_conv(x)
        v = self.conv(x)
        ### Flatten
        q = q.view(q.size()[0], q.size()[1], -1)
        k = k.view(k.size()[0], k.size()[1], -1)
        v = v.view(v.size()[0], v.size()[1], -1)

        k = torch.transpose(k, 1, 2)

        att_filter = torch.bmm(k, q)

        beta = F.softmax(att_filter, dim=-1)
        print('v.shape', v.shape)
        print('beta.shape', beta.shape)

        att_out = torch.bmm(v, beta)
        att_out = att_out.view(x.shape)
        print('att_out.shape', att_out.shape)
        x = x + att_out
        return x.contiguous()


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expand_ratio,
                 reduction=4,  # For Squeeze excitation
                 survival_prob=0.8,
                 ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(hidden_dim / reduction)
        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=1,  # just 1x1 Conv layer to increase #channels
                stride=1, padding=0,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs  # residual connection
        else:
            return self.conv(x)


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


class EfficientNet(nn.Module):
    def __init__(self, version, num_classes, use_attention=False):
        super(EfficientNet, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)  # e.g B1: 1.1, 1.2, 0.2
        last_channel = math.ceil(1280 * width_factor)  # 1280 * 1.1 = 1408
        self.use_attention = use_attention
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.features = self.create_features(width_factor, depth_factor, last_channel)  # e.g B1: 1.1, 1.2, 1408
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes)
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]  # e.g B1 : 1, 260, 0.3
        depth_factor = alpha ** phi  # = 1.2 ** 1
        width_factor = beta ** phi  # = 1.1 ** 1
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):  # e.g B1: 1.1, 1.2, 1408
        channels = int(32 * width_factor)  # 32 * 1.1 = 35
        features = [
            CNNBlock(3, channels, 3, stride=2, padding=1),
        ]  # channels = 35
        in_channels = channels
        # print('in_channels', in_channels)
        # base_model = [
        #     # expand_ratio, channels, repeats, stride, kernel_size
        #     [1, 16, 1, 1, 3],
        #     [6, 24, 2, 2, 3],
        #     [6, 40, 2, 2, 5],
        #     [6, 80, 3, 2, 3],
        #     [6, 112, 3, 1, 5],
        #     [6, 192, 4, 2, 5],
        #     [6, 320, 1, 1, 3]
        for i, (expand_ratio, channels, repeats, stride, kernel_size) in enumerate(base_model):
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
            # print('out_channels', out_channels)
            layers_repeats = math.ceil(repeats * depth_factor)
            # print('layers_repeats', layers_repeats)
            for layer in range(layers_repeats):
                # print('layer', layer)
                # print('in, out', in_channels, out_channels)
                features.append(
                    InvertedResidualBlock(
                        in_channels, out_channels, expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # same padding eg. k=1->pad=0, k=5->pad=2, k=7->pad=3
                    )
                )
                in_channels = out_channels
            # if self.use_attention and i>2:
            #     features.append(SelfAttention(out_channels))
        features.append(CNNBlock(in_channels, last_channels, kernel_size=1, padding=1, stride=1))
        if self.use_attention:
            features.append(SelfAttention(last_channels))
        return nn.Sequential(*features)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(x.view(x.shape[0], -1))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')
    version = 'b0'
    phi, res, drop_rate = phi_values[version]
    num_example, num_classes = 1, 10
    x = torch.rand((num_example, 3, 260, 260))
    model = EfficientNet(
        version=version,
        num_classes=num_classes,
        use_attention=True
    ).to(device)
    print(model(x).shape)
    print(count_parameters(model))
    print(model)

test()
