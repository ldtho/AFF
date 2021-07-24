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
from yacs.config import CfgNode
from src.dataset.preprocessing import Preprocessor


class MURADataset(Dataset):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, node_cfg_dataset: CfgNode, is_training: bool):
        self.data_dir = node_cfg_dataset.path_data_raw
        type = "train" if is_training else "valid"
        img_df = pd.read_csv(join(self.data_dir, f'MURA-v1.1/{type}_image_paths.csv'),
                             names=['path'], header=None)
        label_df = pd.read_csv(join(self.data_dir, f'MURA-v1.1/{type}_labeled_studies.csv'),
                               names=['study', 'label'], header=None)
        img_df = img_df['path'].str.extract('(.+tive\/)(.+)', expand=True)
        img_df.rename(columns={0: 'study', 1: 'image'}, inplace=True)
        df = pd.merge(left=img_df, right=label_df, how='left', on='study')
        df['image_path'] = df['study'] + df['image']
        self.is_training = is_training
        self.df = df[['image_path', 'label']]
        self.imgs = self.df.image_path.values.tolist()
        self.labels = self.df.label.values.tolist()
        self.samples = [tuple(x) for x in self.df.values]
        self.classes = np.unique(self.labels)
        self.balanced_weights = self.balance_class_weights()
        self.transform = Preprocessor(node_cfg_dataset)

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

    def __getitem__(self, idx):
        img_filename = join(self.data_dir, self.imgs[idx])
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        # todo(bdd) : inconsistent right now, need param for grayscale / RGB
        # todo(bdd) : 'L' -> gray, 'RGB' -> Colors
        image = cv2.imread(img_filename)
        # print(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = Image.open(img_filename).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image, is_training=self.is_training)
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


def build_dataloader(data_cfg: CfgNode, is_training: bool) -> DataLoader:
    dataset = MURADataset(node_cfg_dataset=data_cfg, is_training=is_training)
    batch_size = data_cfg.batch_size
    num_worker = min(batch_size, data_cfg.cpu_num)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training,
                            num_workers=num_worker)
    return dataloader


from src.config.config import get_cfg_defaults, combine_cfgs

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    train_loader, = build_dataloader(cfg.dataset, True)
    data_iter = iter(train_loader)
    print(next(data_iter))
