import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import einsum
from BoTnet_CCA import CCA
from collections import OrderedDict
#dcmm

# class Swish(nn.Module):
#     def forward(self, x):
# return x * torch.sigmoid(x)


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
    def __init__(self, block, num_blocks, num_classes=1000, resolution=(224, 224), heads=4, s1=False,
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(512 * block.expansion, 512),
        #     self.activation,
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 1 if num_classes == 2 else num_classes),
        #     # nn.Linear(512, 128),
        #     # self.activation,
        #     # nn.Dropout(0.7),
        #     # nn.Linear(128, 1 if num_classes == 2 else num_classes)
        # )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * block.expansion, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.8),
            nn.Linear(512 * block.expansion, 768, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.8),
            nn.Linear(768, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.8),
            nn.Linear(256, 1 if num_classes == 2 else num_classes))
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
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def BotNet50(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads, s1=s1, cca=cca,
                  silu=silu)


def Resnet50(num_classes=1000, resolution=(224, 224), heads=4):
    return BotNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads,
                  mhsa=False,s1=False, cca=False, silu=False)


def BotNet59_S1(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 6, 6], num_classes=num_classes, resolution=resolution, heads=heads,
                  s1=s1, cca=cca, silu=silu)


def BotNet77_S1(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 6, 12], num_classes=num_classes, resolution=resolution, heads=heads,
                  s1=s1, cca=cca, silu=silu)


def BotNet110_S1(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 23, 6], num_classes=num_classes, resolution=resolution, heads=heads,
                  s1=s1, cca=cca, silu=silu)


def BotNet110_S1_Lite(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, resolution=resolution, heads=heads,
                  s1=s1, cca=cca, silu=silu)


def BotNet128_S1(num_classes=1000, resolution=(224, 224), heads=4, s1=True, cca=True, silu=False):
    return BotNet(BottleNeck, [3, 4, 23, 12], num_classes=num_classes, resolution=resolution, heads=heads,
                  s1=s1, cca=cca, silu=silu)


def load_pretrained_model(model, pretrained_model):
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
                       (k in model_dict) and ('layer4' not in k)}
    # 2. overwrite entries in the existing state dict
    for k in model_dict.keys():
        if k not in pretrained_dict.keys():
            print(k)
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def freeze_pretrained_modules(model, unfreeze_blocks=['layer4', 'avgpool', 'fc'],
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from pytorch_toolbelt.inference import tta

if __name__ == '__main__':
    img_size = 224
    x = torch.randn([2, 3, img_size, img_size])
    model = Resnet50(num_classes=2, heads=8, resolution=(img_size, img_size))

    resnet = torchvision.models.resnet50(pretrained=True)
    model = load_pretrained_model(model, resnet)
    model = freeze_pretrained_modules(model)
    # model2 = BotNet(BottleNeck, [3, 4, 23, 12],  num_classes=2, resolution=(img_size, img_size), heads=8,
    #               s1=True, cca=True,silu=False)
    logits = tta.d4_image2label(model, x)
    print(logits)
    # print(model)
    # model_dict = model.state_dict()
    # resnet_dict = resnet.state_dict()
    # new_resnet_dict = OrderedDict()
    #
    # resnet_keys = list(resnet_dict.keys())
    # for idx, item in enumerate(resnet_keys):
    #     if "conv2.weight" in item:
    #         resnet_keys[idx] = item.replace("conv2.weight", 'conv2.0.weight')
    #     if 'downsample.1' in item:
    #         resnet_keys[idx] = item.replace("downsample.1", 'downsample.2')
    #     if 'downsample.0' in item:
    #         resnet_keys[idx] = item.replace("downsample.0", 'downsample.1')
    # for key, value in zip(resnet_keys, resnet_dict.values()):
    #     new_resnet_dict[key] = value
    # pretrained_dict = {k: v for k, v in new_resnet_dict.items() if (k in model_dict) and ('layer4' not in k) and ('fc.1' not in k)}
    #
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(pretrained_dict,strict=False)
    # print(model.state_dict())
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #
    # print(model_dict.keys())
    # print(resnet.state_dict().keys())
    #
    # print(model)
    # print('=================================')
    # print([x for x in list(resnet.state_dict())])
    # print(resnet)

# def pair(x):
#     return (x, x) if not isinstance(x, tuple) else x
#
#
# def expand_dim(t, dim, k):
#     t = t.unsqueeze(dim=dim)
#     expand_shape = [-1] * len(t.shape)
#     expand_shape[dim] = k
#     return t.expand(*expand_shape)
#
#
# def rel_to_abs(x):
#     b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
#     # print('b, h, l, _, device, dtype',b, h, l, _, device, dtype)
#     dd = {'device': device, 'dtype': dtype}
#     col_pad = torch.zeros((b, h, l, 1), **dd)
#     x = torch.cat((x, col_pad), dim=3)
#     # print('x.shape',x.shape)
#     flat_x = rearrange(x, 'b h l c -> b h (l c)')
#     flat_pad = torch.zeros((b, h, l - 1), **dd)
#     # print('flat_pad.shape',flat_pad.shape)
#     flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
#     # print('flat_x_padded.shape',flat_x_padded.shape)
#     final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
#     # print('final_x.shape',final_x.shape)
#     final_x = final_x[:, :, :l, (l - 1):]
#     # print('final_x.shape2',final_x.shape)
#     return final_x
#
#
# def relative_logits_1d(q, rel_k):
#     b, heads, h, w, dim = q.shape
#     logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
#     logits = rearrange(logits, 'b h x y r -> b (h x) y r')
#     logits = rel_to_abs(logits)
#     logits = logits.reshape(b, heads, h, w, w)
#     # print('logits.shape',logits.shape)
#     logits = expand_dim(logits, dim=3, k=h)
#     # print('logits.shape2',logits.shape)
#     return logits
#
#
# # positional embeddings
#
# class AbsPosEmb(nn.Module):
#     def __init__(self, fmap_size, dim_head):
#         super().__init__()
#         height, width = pair(fmap_size)
#         scale = dim_head ** -0.5
#         self.height = nn.Parameter(torch.randn([height, dim_head]) * scale, requires_grad=True)
#         self.width = nn.Parameter(torch.randn([width, dim_head]) * scale, requires_grad=True)
#
#     def forward(self, q):
#         print('q.shape',q.shape)
#         emb_h, emb_w = rearrange(self.height, 'h d -> h () d'),rearrange(self.width, 'w d -> () w d')
#         print('emb_h.shape',emb_h.shape)
#         print('emb_w.shape',emb_w.shape)
#         emb =  emb_h + emb_w
#
#         emb = rearrange(emb, ' h w d -> (h w) d')
#         logits = einsum('b h i d, j d -> b h i j', q, emb)
#         return logits
#
#
# class RelPosEmb(nn.Module):
#     def __init__(self, fmap_size, dim_head):
#         super().__init__()
#         height, width = pair(fmap_size)
#         scale = dim_head ** -0.5
#         self.fmap_size = fmap_size
#         self.rel_height = nn.Parameter(torch.randn([int(height * 2 - 1), dim_head]) * scale, requires_grad=True)
#         self.rel_width = nn.Parameter(torch.randn([int(width * 2 - 1), dim_head]) * scale, requires_grad=True)
#
#     def forward(self, q):
#         h, w = self.fmap_size
#         print('self.fmap_size',self.fmap_size)
#
#         q = rearrange(q, 'b h (x y) d -> b h x y d', x=h, y=w)
#         rel_logits_w = relative_logits_1d(q, self.rel_width)
#         print('q.shape, self.rel_width.shape',q.shape, self.rel_width.shape)
#         rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')
#
#         q = rearrange(q, 'b h x y d -> b h y x d')
#         rel_logits_h = relative_logits_1d(q, self.rel_height)
#         rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
#         return rel_logits_w + rel_logits_h
#
#
# # classes
#
# class MHSA(nn.Module):
#     def __init__(self, *, dim, fmap_size, heads=4, dim_head=128, rel_pos_emb=False):
#         super(MHSA, self).__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         inner_dim = heads * dim_head
#         # print('inner_dim',inner_dim)
#
#         self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
#
#         rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
#         self.pos_emb = rel_pos_class(fmap_size, dim_head)
#
#     def forward(self, fmap):
#         heads, b, c, h, w = self.heads, *fmap.shape
#         print("heads, b, c, h, w", heads, b, c, h, w)
#
#         q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
#         # print('q.shape', q.shape, 'k.shape', k.shape, 'v.shape', v.shape)
#         q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))
#         # print('q.shape2', q.shape, 'k.shape2', k.shape, 'v.shape2', v.shape)
#         # print('self.scale,q*self.scale', self.scale, (q * self.scale).shape)
#         q *= self.scale
#
#         sim = einsum('b h i d, b h j d -> b h i j', q, k)
#         # print('sim.shape', sim.shape)
#         sim += self.pos_emb(q)
#         print('sim.shape, q.shape ,self.pos_emb(q).shape', sim.shape, q.shape, self.pos_emb(q).shape)
#         attn = sim.softmax(dim=-1)
#
#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
#         return out
