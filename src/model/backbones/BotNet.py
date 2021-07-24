import torch
import torch.nn as nn
import torchvision.models
from torch import einsum
from .BoTnet_CCA import CCA
from collections import OrderedDict
from src.model.builder.backbone_builder import BACKBONE_REGISTRY
from src.config.config import get_cfg_defaults
from yacs.config import CfgNode

botnet_arch = {
    'botnet50': [3, 4, 6, 3],
    'botnet59': [3, 4, 6, 6],
    'botnet77': [3, 4, 6, 12],
    'botnet110': [3, 4, 23, 6],
    'botnet128': [3, 4, 23, 12],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 6]
}
resnet_equivalent = {
    'botnet50': torchvision.models.resnet50,
    'botnet59': torchvision.models.resnet50,
    'botnet77': torchvision.models.resnet50,
    'botnet110': torchvision.models.resnet101,
    'botnet128': torchvision.models.resnet101,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101
}



class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        self.head_dims = n_dims // heads
        self.scale = self.head_dims ** -0.5
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.relative_h = nn.Parameter(torch.randn([self.head_dims, height, 1]) * self.scale,
                                       requires_grad=True)
        self.relative_w = nn.Parameter(torch.randn([self.head_dims, 1, width]) * self.scale,
                                       requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [1,1280,10,11]
        # print('x.shape',x.shape)
        n_batch, C, height, width = x.shape
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)  # 1,4,320,110
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)  # 1,4,320,110
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)  # 1,4,320,110

        q *= self.scale

        content_content = einsum('b h d i, b h d j -> b h i j', q, k)
        # content_content = torch.matmul(k.permute(0, 1, 3, 2), q)  # 1,4,110,110

        content_position = (self.relative_w + self.relative_h)
        content_position = content_position.view(self.head_dims, -1)
        content_position = einsum('b h d i, d j -> b h i j', q, content_position)

        energy = content_content + content_position  # 1,4,110,110
        beta = self.softmax(energy)

        attention = einsum('b h i j, b h d j -> b h i d', beta, v)
        attention = attention.view(x.shape)
        return attention


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None, s1=False,
                 cca=False, silu=False):
        super(BottleNeck, self).__init__()
        self.cca = cca
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ModuleList()
        self.activation = nn.SiLU(inplace=True) if silu else nn.ReLU(inplace=True)
        # print(resolution)
        if not mhsa:
            self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
            if cca:
                self.conv2.append(
                    CCA(planes, 8, pool_types=['avg', 'max'], channel_att=False, spatial_att=False, coord_att=True,
                        silu=silu))
        else:
            self.conv2.append(
                MHSA(planes, height=int(resolution[0]), width=int(resolution[1]), heads=heads)
                # MHSA(dim=planes, fmap_size=(int(resolution[0]), int(resolution[1])), heads=4, dim_head=128,
                #      rel_pos_emb=False)
            )
            if stride == 2:
                self.conv2.append((nn.AvgPool2d(2, 2)))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            if not s1:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d((2, 2)),
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.activation(out)
        return out


class BotNet(nn.Module):
    def __init__(self, block, num_blocks, resolution=(224, 224), heads=4, s1=False,
                 cca=False, mhsa=True, silu=False):
        super(BotNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resolution = [self.resolution[0] / self.conv1.stride[0], self.resolution[1] / self.conv1.stride[1]]
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.SiLU(inplace=True) if silu else nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2, silu=silu)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cca=cca, silu=silu)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cca=cca, silu=silu)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1 if s1 else 2, heads=heads,
                                       mhsa=mhsa, s1=s1, silu=silu)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.BatchNorm1d(512 * block.expansion, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Dropout(p=0.8),
        #     nn.Linear(512 * block.expansion, 768, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Dropout(p=0.8),
        #     nn.Linear(768, 256, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Dropout(p=0.8),
        #     nn.Linear(256, 1 if num_classes == 2 else num_classes))
        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        def init(m):
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    # Fills the input Tensor with values according to the method described in Delving deep into
                    # rectifiers: Surpassing human-level performance on
                    # ImageNet classification - He, K. et al. (2015), using a normal distribution
                    init_fn(child.weights)

        init(self)

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False, s1=False, cca=False, silu=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            #       in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution, s1, cca, silu))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print('out.shape', out.shape)
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        # out = self.fc(out)
        return out


def _load_imagenet_weights(model, pretrained_model):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_model.state_dict()
    new_pretrained_dict = OrderedDict()
    pretrained_keys = list(pretrained_model_dict.keys())
    for idx, item in enumerate(pretrained_keys):
        if "conv2.weight" in item:
            pretrained_keys[idx] = item.replace("conv2.weight", 'conv2.0.weight')
        if 'downsample.1' in item:
            pretrained_keys[idx] = item.replace("downsample.1", 'downsample.2')
        if 'downsample.0' in item:
            pretrained_keys[idx] = item.replace("downsample.0", 'downsample.1')
    for key, value in zip(pretrained_keys, pretrained_model_dict.values()):
        new_pretrained_dict[key] = value

    pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if
                       (k in model_dict)
                       and ('layer4' not in k)
                       }
    # 2. overwrite entries in the existing state dict
    for k in model_dict.keys():
        if k not in pretrained_dict.keys():
            print(k)
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def _freeze_pretrained_modules(model, unfreeze_blocks=['layer4', 'avgpool', 'fc'],
                               unfreeze_modules=['ChannelGate', 'CoordAtt', 'bn', 'activation']):
    for name, child in model.named_children():
        if name in unfreeze_blocks:
            # print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            # print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
    for (name, param) in model.named_parameters():
        if any(x in name for x in unfreeze_modules):
            param.requires_grad = True
            print(name, 'is unfrozen')

    print("frozen layers list:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name)
    return model


def _check_pretrained(backbone_cfg: CfgNode, model) -> nn.Module:
    if backbone_cfg.pretrained_path:
        model.load_state_dict(torch.load(backbone_cfg.pretrained_path))
        return model
    elif backbone_cfg.load_imagenet_weights:
        resnet = resnet_equivalent[backbone_cfg.version](pretrained=True)
        model = _load_imagenet_weights(model, pretrained_model=resnet)
    if all([x for x in (backbone_cfg.unfreeze_blocks, backbone_cfg.unfreeze_modules)]):
        model = _freeze_pretrained_modules(model,unfreeze_blocks=backbone_cfg.unfreeze_blocks,
                                           unfreeze_modules=backbone_cfg.unfreeze_modules)
    return model


@BACKBONE_REGISTRY.register('botnet')
def _BotNet(backbone_cfg: CfgNode) -> nn.Module:
    print(backbone_cfg.input_res)
    model = BotNet(BottleNeck,botnet_arch[backbone_cfg.version],
                   resolution=backbone_cfg.input_res,
                   heads=backbone_cfg.num_heads, s1=backbone_cfg.use_S1, cca=backbone_cfg.use_CCA,
                   mhsa=True, silu=backbone_cfg.silu
                   )
    model = _check_pretrained(backbone_cfg, model)
    return model


@BACKBONE_REGISTRY.register('resnet')
def _ResNet(backbone_cfg: CfgNode) -> nn.Module:
    model = BotNet(BottleNeck, botnet_arch[backbone_cfg.version], resolution=backbone_cfg.input_res,
                   heads=backbone_cfg.num_heads,
                   mhsa=False, s1=False, cca=False, silu=False)
    model = _check_pretrained(backbone_cfg, model)
    return model


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    backbone_cfg = cfg.model.backbone
    model = BACKBONE_REGISTRY['botnet'](backbone_cfg)
    x = torch.rand((2,3,320,320))
    print(model(x).shape)
