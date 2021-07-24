from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import re
import os
import torch
import numbers
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from os.path import join
from PIL import Image
import cv2
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# pretrained_size = 320
#
# pretrained_means = [0.485,0.456,0.406]
#
# pretrained_stds= [0.229,0.224,0.225]
DATA_DIR = '/home/starlet/data'
data_cat = ['train', 'valid']


class MURADataset(Dataset):
    url = 'https://cs.stanford.edu/group/mlgroup/mura-v1.1.zip'
    filename = 'MURA-v1.1.ZIP'
    md5_checksum = '4c36feddb7f5698c8bf291b912c438b1'
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, data_dir=DATA_DIR, transform=None, train=True):
        self.data_dir = data_dir
        type = 'train' if train else 'valid'
        img_df = pd.read_csv(join(data_dir, f'MURA-v1.1/{type}_image_paths.csv'), names=['path'], header=None)
        label_df = pd.read_csv(join(data_dir, f'MURA-v1.1/{type}_labeled_studies.csv'), names=['study', 'label'],
                               header=None)
        img_df = img_df['path'].str.extract('(.+tive\/)(.+)', expand=True)
        img_df.rename(columns={0: 'study', 1: 'image'}, inplace=True)
        df = pd.merge(left=img_df, right=label_df, how='left', on='study')
        df['image_path'] = df['study'] + df['image']
        # print(df.label.value_counts())
        self.df = df[['image_path', 'label']]
        self.imgs = self.df.image_path.values.tolist()
        self.labels = self.df.label.values.tolist()
        self.samples = [tuple(x) for x in self.df.values]
        self.classes = np.unique(self.labels)
        self.balanced_weights = self.balance_class_weights()
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def balance_class_weights(self):
        count = [0] * len(self.classes)
        for item in self.samples:
            count[item[1]] += 1
        weight_per_class = [0.] * len(self.classes)
        N = float(sum(count))
        for i in range(len(self.classes)):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(self.samples)
        for idx, val in enumerate(self.samples):
            weight[idx] = weight_per_class[val[1]]
        return weight

    def __getitem__(self, idx):
        img_filename = join(self.data_dir, self.imgs[idx])
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        # todo(bdd) : inconsistent right now, need param for grayscale / RGB
        # todo(bdd) : 'L' -> gray, 'RGB' -> Colors
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }
        return image, label, meta_data


class SquarePad:
    def __call__(self, image):
        c, w, h = image.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, value=0, mode='constant')


# now use it as the replacement of transforms.Pad class


def get_dataloaders(data_dir, image_size, batch_size=8, num_worker = 2):
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.ToTensor(),
    #         SquarePad(),
    #         transforms.ToPILImage(),
    #         transforms.Resize((image_size, image_size)),
    #         transforms.RandomChoice([
    #             transforms.ColorJitter(brightness=0.5),
    #             transforms.ColorJitter(contrast=0.5),
    #             transforms.ColorJitter(saturation=0.5),
    #             transforms.ColorJitter(hue=0.5),
    #             transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    #             transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    #             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #         ]),
    #         transforms.RandomChoice([
    #             transforms.RandomRotation((0, 0)),
    #             transforms.RandomHorizontalFlip(p=1),
    #             transforms.RandomVerticalFlip(p=1),
    #             transforms.RandomRotation((90, 90)),
    #             transforms.RandomRotation((180, 180)),
    #             transforms.RandomRotation((270, 270)),
    #             transforms.Compose([
    #                 transforms.RandomHorizontalFlip(p=1),
    #                 transforms.RandomRotation((90, 90)),
    #             ]),
    #             transforms.Compose([
    #                 transforms.RandomHorizontalFlip(p=1),
    #                 transforms.RandomRotation((270, 270)),
    #             ])
    #         ]),
    #         # transforms.RandomResizedCrop(),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'valid': transforms.Compose([
    #         transforms.ToTensor(),
    #         SquarePad(),
    #         transforms.ToPILImage(),
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    # }
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size,image_size,border_mode=cv2.BORDER_CONSTANT,value=0),
            A.Flip(p=0.7),
            A.HorizontalFlip(p=0.7),
            A.Transpose(p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.OneOf([
                A.MotionBlur(blur_limit=5,p=1),
                A.MedianBlur(blur_limit=5,p=1),
                A.GaussianBlur(blur_limit=5,p=1),
                A.GaussNoise(var_limit=(5.0, 30.0),p=1),
            ], p=0.7),
            A.CLAHE(clip_limit=4.0, p=0.7),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.3, rotate_limit=30, p=0.7),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(image_size, image_size,border_mode=cv2.BORDER_CONSTANT,value=0),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_dataset = MURADataset(data_dir, train_transform, train=True)

    val_dataset = MURADataset(data_dir, val_transform, train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_worker)

    return train_dataset, val_dataset, train_dataloader, valid_dataloader


if __name__ == '__main__':
    train_dataset, val_dataset, train_dataloader, valid_dataloader = get_dataloaders(DATA_DIR, image_size=224,
                                                                                     batch_size=4)
    print(next(iter(train_dataloader))[0].shape)
