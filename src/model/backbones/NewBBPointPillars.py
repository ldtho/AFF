import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import einsum
from BoTnet_CCA import CCA
from collections import OrderedDict


class MHSA(nn.Module):
    def __init__(self, n_dims, height=14, width=14, heads=4):
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

        content_position = (self.relative_w + self.relative_h)
        content_position = content_position.view(self.head_dims, -1)
        content_position = einsum('b h d i, d j -> b h i j', q, content_position)

        energy = content_content + content_position
        beta = self.softmax(energy)

        attention = einsum('b h i j, b h d j -> b h i d', beta, v)
        attention = attention.view(x.shape)
        return attention


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None,
                 cca=True):
        super(BottleNeck, self).__init__()
        self.cca = cca
        self.mhsa = mhsa
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = []
        if not mhsa:
            self.conv2.append(nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                                        stride=stride, bias=False))
            if cca:
                self.conv2.append(CCA(planes, 8, pool_types=['avg', 'max'], channel_att=False,
                                      spatial_att=False, coord_att=True))
        else:
            self.conv2.append(
                MHSA(planes, height=int(resolution[0]), width=int(resolution[1]),
                     heads=heads)
            )
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
        self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class BotNet_RPN(nn.Module):  # input size: batch, C, width, length [1,64,400,600]
    def __init__(self, block,
                 num_blocks=[3, 5, 5],
                 mid_planes=[64, 64, 128],
                 num_class=1000,
                 resolution=[400, 400],
                 heads=4,
                 cca=True,
                 mhsa=True,
                 strides=[2, 2, 2],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=False,
                 use_bev=False,
                 box_code_size=7,
                 ):
        super(BotNet_RPN, self).__init__()
        self.resolution = list(resolution)
        self.in_planes = 64
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self.conv1 = nn.Conv2d(num_input_filters, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.resolution = [self.resolution[0] / self.conv1.stride[0],
                           self.resolution[1] / self.conv1.stride[1]]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, mid_planes[0], num_blocks[0],
                                       stride=strides[0], heads=heads, mhsa=False,
                                       cca=cca)
        print(mid_planes[0] * 4,
              mid_planes[2] * 4,
              upsample_strides[0],
              upsample_strides[0])
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[0] * 4,
                               mid_planes[2] * 4,
                               upsample_strides[0],
                               stride=upsample_strides[0]
                               ),
            nn.BatchNorm2d(mid_planes[2] * 4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.layer2 = self._make_layer(block, mid_planes[1], num_blocks[1],
                                       stride=strides[1], heads=heads, mhsa=False,
                                       cca=cca)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[1] * 4,
                               mid_planes[2] * 4,
                               upsample_strides[1],
                               stride=upsample_strides[1]
                               ),
            nn.BatchNorm2d(mid_planes[2] * 4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.layer3 = self._make_layer(block, mid_planes[2], num_blocks[2],
                                       stride=strides[2], heads=heads, mhsa=mhsa,
                                       cca=False)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(mid_planes[2] * 4,
                               mid_planes[2] * 4,
                               upsample_strides[2],
                               stride=upsample_strides[2]
                               ),
            nn.BatchNorm2d(mid_planes[2] * 4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(mid_planes[-1] * 4 * len(num_blocks), num_cls, kernel_size=1)
        self.conv_box = nn.Conv2d(mid_planes[-1] * 4 * len(num_blocks), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(mid_planes[-1] * 4 * len(num_blocks), num_anchor_per_loc * 2, kernel_size=1)

    def _make_layer(self, block, planes, num_block, stride=1, heads=4, mhsa=False,
                    cca=False):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa,
                                self.resolution, cca))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        up1 = self.deconv1(x)
        x = self.layer2(x)
        up2 = self.deconv2(x)
        x = self.layer3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_tuple = (box_preds, cls_preds)
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            # ret_dict["dir_cls_preds"] = dir_cls_preds
            ret_tuple += (dir_cls_preds,)
        return ret_tuple


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    x = torch.rand([1, 64, 400, 400])
    model = BotNet_RPN(BottleNeck,
                       num_blocks=[3, 5, 5],
                       mid_planes=[64, 64, 128],
                       num_class=1000,
                       resolution=[400, 400],
                       heads=4, cca=True,
                       mhsa=True,
                       strides=[2, 2, 2],
                       upsample_strides=[1, 2, 4],
                       num_upsample_filters=[256, 256, 256],
                       num_input_filters=64,
                       num_anchor_per_loc=2,
                       encode_background_as_zeros=True,
                       use_direction_classifier=False,
                       use_bev=False,
                       box_code_size=7)
    print(count_parameters(model))
    print(model(x))
