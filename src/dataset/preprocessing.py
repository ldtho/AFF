from __future__ import absolute_import, division, print_function
from typing import Union, Tuple
import cv2
import numpy as np
from albumentations import LongestMaxSize, PadIfNeeded, Flip, HorizontalFlip, \
    Transpose, RandomBrightnessContrast, OneOf, MotionBlur, MedianBlur, GaussNoise, CLAHE, Compose, RGBShift, \
    ShiftScaleRotate, Resize, Normalize, Blur, GridDropout, \
    GridDistortion
from albumentations.pytorch import ToTensorV2
from yacs.config import CfgNode
from .augmix import augmentations, augment_and_mix
from .augmix.augment_and_mix import RandomAugMix

class Preprocessor():
    def __init__(self, node_cfg_dataset: CfgNode):
        self.aug_cfg = node_cfg_dataset.augmentation
        self.color_aug = self.generate_color_augmentation(self.aug_cfg)
        self.shape_aug = self.generate_shape_augmentation(self.aug_cfg)
        self.noise_aug = self.generate_noise_augmentation(self.aug_cfg)
        img_size = self.aug_cfg.image_size
        self.pad_resize = Compose([
            LongestMaxSize(max_size=img_size),
            PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]) if self.aug_cfg.keep_ratio else None
        self.resize = Resize(height=img_size, width=img_size) if self.aug_cfg.resize else None
        self.normalize = Normalize(self.aug_cfg.normalize_mean,
                                   self.aug_cfg.normalize_std) if self.aug_cfg.normalize else None
        self.to_tensor = ToTensorV2()

        self.do_augmix = node_cfg_dataset.do_augmix
        if self.do_augmix:
            augmentations.IMAGE_SIZE = img_size
            self.augmix = RandomAugMix(severity=self.aug_cfg.augmix_severity,
                                       width=self.aug_cfg.augmix_width,
                                       alpha=self.aug_cfg.augmix_alpha,
                                       p=self.aug_cfg.augmix_prop)

    @staticmethod
    def generate_noise_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        nosie_aug_list = []
        if aug_cfg.blur_prop > 0:
            blurring = OneOf([
                MotionBlur(aug_cfg.blur_limit, p=1),
                MedianBlur(aug_cfg.blur_limit, p=1),
                Blur(aug_cfg.blur_limit, p=1)
            ], p=aug_cfg.blur_prop)
            nosie_aug_list.append(blurring)
        if aug_cfg.gaussian_noise_prop > 0:
            nosie_aug_list.append(GaussNoise(aug_cfg.gaussian_noise_limit,
                                             p=aug_cfg.gaussian_noise_prop))
        if aug_cfg.gridmask_prop > 0:
            nosie_aug_list.append(GridDropout(aug_cfg.gridmask_ratio,
                                              random_offset=aug_cfg.gridmask_random_offset,
                                              p=aug_cfg.gridmask_ratio))
        if len(nosie_aug_list) > 0:
            return Compose(nosie_aug_list)
        else:
            return None

    @staticmethod
    def generate_shape_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        shape_aug_list = []
        if aug_cfg.shiftscalerotate_prop > 0:
            shape_aug_list.append(
                ShiftScaleRotate(shift_limit=aug_cfg.shift_limit,
                                 scale_limit=aug_cfg.scale_limit,
                                 rotate_limit=aug_cfg.rotate_limit)
            )
        if aug_cfg.grid_distortion_prop > 0:
            shape_aug_list.append(GridDistortion(aug_cfg.grid_distortion_prop))
        if aug_cfg.flip_prop:
            shape_aug_list.append(Flip(p=aug_cfg.flip_prop))
        if aug_cfg.horizontal_flip_prop > 0:
            shape_aug_list.append(HorizontalFlip(p=aug_cfg.horizontal_flip_prop))
        if aug_cfg.transpose_prop > 0:
            shape_aug_list.append(Transpose(p=aug_cfg.transpose_prop))
        if len(shape_aug_list) > 0:
            return Compose(shape_aug_list, p=1)
        else:
            return None

    @staticmethod
    def generate_color_augmentation(aug_cfg: CfgNode) -> Union[Compose, None]:
        color_aug_list = []
        if aug_cfg.brightness_contrast_prop > 0:
            color_aug_list.append(RandomBrightnessContrast(p=aug_cfg.brightness_contrast_prop))
        if aug_cfg.CLAHE_prop > 0:
            color_aug_list.append(CLAHE(clip_limit=aug_cfg.CLAHE_limit,
                                        p=aug_cfg.CLAHE_prop))
        if aug_cfg.rgb_shift_prop > 0:
            color_aug_list.append(RGBShift(r_shift_limit=aug_cfg.rgb_shift_red,
                                           g_shift_limit=aug_cfg.rgb_shift_green,
                                           b_shift_limit=aug_cfg.rgb_shift_blue,
                                           p=aug_cfg.rgb_shift_prop))
        if len(color_aug_list) > 0:
            return Compose(color_aug_list, p=1)
        else:
            return None

    def __call__(self, img: np.ndarray, is_training=bool, normalize: bool = True) -> Union[np.ndarray, Tuple]:
        x = img
        x = self.pad_resize(image=x)['image'] if self.pad_resize is not None else x
        if is_training:
            if self.do_augmix:
                x = self.augmix(image = x)['image']
            else:
                x = self.color_aug(image=x)['image'] if self.color_aug is not None else x
                x = self.noise_aug(image=x)['image'] if self.noise_aug is not None else x
                x = self.shape_aug(image=x)['image'] if self.shape_aug is not None else x
        x = self.resize(image=x)['image'] if self.resize is not None else x
        x = self.normalize(image=x)['image'] if self.normalize is not None else x
        x = self.to_tensor(image=x)['image']
        return x



