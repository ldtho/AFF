import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=False,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 ):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self.block1 = [
            nn.Conv2d(num_input_filters, num_filters[0], 3, layer_strides[0], padding=1, bias=False),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU()
        ]
        for i in range(layer_nums[0]):
            self.block1 += [
                nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[0], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
        self.block1 = nn.Sequential(*self.block1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[0],
                               num_upsample_filters[0],
                               upsample_strides[0],
                               stride=upsample_strides[0]),
            nn.BatchNorm2d(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.block2 = [
            nn.Conv2d(num_filters[0], num_filters[1], 3, padding=1,
                      stride=layer_strides[1], bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU()
        ]
        for i in range(layer_nums[2]):
            self.block2 += [
                nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[1]),
                nn.ReLU()
            ]
        self.block2 = nn.Sequential(*self.block2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[1],
                               num_upsample_filters[1],
                               upsample_strides[1],
                               stride=upsample_strides[1]
                               ),
            nn.BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU()
        )
        self.block3 = [
            nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2],
                      padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU()
        ]
        for i in range(layer_nums[2]):
            self.block3 += [
                nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters[2]),
                nn.ReLU()
            ]
        self.block3 = nn.Sequential(*self.block3)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(num_filters[2],
                               num_upsample_filters[2],
                               upsample_strides[2],
                               stride=upsample_strides[2]),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters),
                                  num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1
            )


    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        print('box_preds.shape',box_preds.shape)
        print('cls_preds.shape',cls_preds.shape)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        print('box_preds.shape',box_preds.shape)
        print('cls_preds.shape',cls_preds.shape)
        ret_tuple = (box_preds, cls_preds)
        return ret_tuple


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def RPNCifar(num_class):
    return RPN(num_class=num_class,
               layer_nums=[3, 5, 5],
               layer_strides=[2, 2, 2],
               num_filters=[128, 128, 256],
               upsample_strides=[1, 2, 4],
               num_upsample_filters=[256, 256, 256],
               num_input_filters=128,
               num_anchor_per_loc=2,
               encode_background_as_zeros=True,
               use_direction_classifier=False)


if __name__ == '__main__':
    model = RPN(num_class=10,
                layer_nums=[3, 5, 5],
                layer_strides=[2, 2, 2],
                num_filters=[128, 128, 256],
                upsample_strides=[1, 2, 4],
                num_upsample_filters=[256, 256, 256],
                num_input_filters=128,
                num_anchor_per_loc=2,
                encode_background_as_zeros=True,
                use_direction_classifier=False,
                box_code_size=4)
    device = torch.device('cpu')
    x = torch.randn(1, 128, 16, 16)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
    print(count_parameters(model))
    print(y[0])
