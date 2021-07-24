import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url
import torchvision

########################################
######## INPUT SIZE = (224*224) ########
########################################

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class VGG(nn.Module):
    def __init__(self,
                 features: nn.Module,
                 num_classes: int = 1000,
                 init_weights: bool = True) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x

    def _initialize_weights(self):
        print(tuple(self.modules()))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            # layers += [SE_Block(v)]
            in_channels = v
        # layers += [PrintSize()]

    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "E": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = model.state_dict()
        param_names = list(state_dict.keys())
        # print(state_dict)
        pretrained_state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        pretrained_param_names = list(pretrained_state_dict.keys())

        print(param_names)
        print(pretrained_param_names)
        for i, param in enumerate(param_names):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        model.load_state_dict(state_dict)
        print('\n Loaded base model')
    return model


def vgg16(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg16', cfg="D", batch_norm=False, pretrained=pretrain, progress=progress, **kwargs)


def vgg16_bn(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg16_bn', cfg="D", batch_norm=True, pretrained=pretrain, progress=progress, **kwargs)


def vgg19(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg19', cfg="E", batch_norm=False, pretrained=pretrain, progress=progress, **kwargs)


def vgg19_bn(pretrain: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg('vgg19_bn', cfg="E", batch_norm=True, pretrained=pretrain, progress=progress, **kwargs)
