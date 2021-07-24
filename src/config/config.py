import os
from pathlib import Path
from yacs.config import CfgNode as ConfigurationNode
from dotenv import load_dotenv, find_dotenv

from albumentations.pytorch import ToTensorV2

__C = ConfigurationNode()

__C.output_path = '/kaggle/output/AFF'
__C.multi_gpu_training = False
__C.device = 'cpu'
__C.resume_path = ''
# data augmentation parameters with albumentations lib
__C.dataset = ConfigurationNode()
# Data path
__C.dataset.train_data_path = '/home/starlet/data/MURA-v1.1/train'
__C.dataset.val_data_path = "/home/starlet/data/MURA-v1.1/valid"
__C.dataset.path_data_raw = '/home/starlet/data'
# Dataloader
__C.dataset.batch_size = 2
__C.dataset.cpu_num = 4  # num_worker = min(batch_size, data_cfg.cpu_num)

# AUGMENTATION
__C.dataset.augmentation = ConfigurationNode()
# AugMIX
__C.dataset.do_augmix = False
__C.dataset.augmentation.augmix_severity = 4
__C.dataset.augmentation.augmix_width = 3
__C.dataset.augmentation.augmix_alpha = 1.0
__C.dataset.augmentation.augmix_prop = 0.7
# Default values
__C.dataset.augmentation.resize = True
__C.dataset.augmentation.keep_ratio = True
__C.dataset.augmentation.image_size = 320
# shape aug
__C.dataset.augmentation.flip_prop = 0.5
__C.dataset.augmentation.horizontal_flip_prop = 0.5
__C.dataset.augmentation.transpose_prop = 0.5
__C.dataset.augmentation.grid_distortion_prop = 0.5
__C.dataset.augmentation.shiftscalerotate_prop = 0.5
__C.dataset.augmentation.shift_limit = 0.05
__C.dataset.augmentation.scale_limit = 0.3
__C.dataset.augmentation.rotate_limit = 30  # degree
# noise and blur aug
__C.dataset.augmentation.blur_prop = 0.5
__C.dataset.augmentation.blur_limit = 5
__C.dataset.augmentation.gaussian_blur_limit = 5
__C.dataset.augmentation.gaussian_noise_prop = 0.5
__C.dataset.augmentation.gaussian_noise_limit = (5.0, 30.0)
__C.dataset.augmentation.gridmask_prop = 0.5
__C.dataset.augmentation.gridmask_ratio = 0.3
__C.dataset.augmentation.gridmask_random_offset = False
# color aug
__C.dataset.augmentation.brightness_contrast_prop = 0.5
__C.dataset.augmentation.CLAHE_prop = 0.5
__C.dataset.augmentation.CLAHE_limit = 0.7
__C.dataset.augmentation.rgb_shift_prop = 0.5
__C.dataset.augmentation.rgb_shift_red = 15
__C.dataset.augmentation.rgb_shift_green = 15
__C.dataset.augmentation.rgb_shift_blue = 15
# normalize
__C.dataset.augmentation.normalize = True
__C.dataset.augmentation.normalize_mean = (0.485, 0.456, 0.406)
__C.dataset.augmentation.normalize_std = (0.229, 0.224, 0.225)

# Model backbone config
__C.model = ConfigurationNode()
__C.model.parallel = False
__C.model.meta_architecture = 'baseline'
# __C.model.normalization_fn = 'BN'

__C.model.backbone = ConfigurationNode()
__C.model.backbone.name = 'botnet'
__C.model.backbone.input_res = (__C.dataset.augmentation.image_size, __C.dataset.augmentation.image_size)
# For BotNet
__C.model.backbone.version = 'botnet50'
__C.model.backbone.use_CCA = True
__C.model.backbone.use_S1 = True
__C.model.backbone.silu = True
__C.model.backbone.num_heads = 4
__C.model.backbone.load_imagenet_weights = False
__C.model.backbone.unfreeze_blocks = ['layer4', 'avgpool', 'fc']
__C.model.backbone.unfreeze_modules = ['ChannelGate', 'CoordAtt', 'bn', 'activation']

__C.model.backbone.pretrained_path = False

# Model head configs
__C.model.head = ConfigurationNode()
__C.model.head.name = 'simple_classification_head'
__C.model.head.activation = 'relu'
__C.model.head.batch_norm = True
__C.model.head.input_dims = 2048
__C.model.head.hidden_dims = [512, 256]
__C.model.head.output_dims = 1
__C.model.head.dropout = 0.4

# Optimizer
__C.model.solver = ConfigurationNode()
__C.model.solver.use_amp = False
__C.model.solver.optimizer = ConfigurationNode()
__C.model.solver.optimizer.base_lr = 0.001
__C.model.solver.optimizer.name = 'adam'
__C.model.solver.optimizer.weight_decay = 1e-4

# SGD config
__C.model.solver.optimizer.SGD = ConfigurationNode()
__C.model.solver.optimizer.SGD.momentum = 0.9
__C.model.solver.optimizer.SGD.nesterov = False

# Scheduler
# options multi_steps/reduce_on_plateau/one_cycle/cos_anneal_warmup_restart
__C.model.solver.scheduler = ConfigurationNode()
__C.model.solver.scheduler.name = 'unchange'
__C.model.solver.scheduler.max_lr = 1e-4

# MultiStep and ReduceOnPlateau
__C.model.solver.scheduler.lr_reduce_gamma = 0.1
# ReductOnPlateau
__C.model.solver.scheduler.patience = 5
# MultiStepLR hyperparams
__C.model.solver.scheduler.multi_steps_lr_milestones = []

# OneCycleLR hyperparams
__C.model.solver.scheduler.one_cycle = ConfigurationNode()
__C.model.solver.scheduler.one_cycle.pct_start = 0.3
__C.model.solver.scheduler.one_cycle.anneal_strategy = 'cos'
__C.model.solver.scheduler.one_cycle.div_factor = 30
__C.model.solver.scheduler.one_cycle.total_epochs = 40
__C.model.solver.scheduler.one_cycle.cycle_momentum = True
__C.model.solver.scheduler.one_cycle.steps_per_epoch = 1841

# CosineAnnealingWarmupRestarts hyperparams
__C.model.solver.scheduler.cos_anneal = ConfigurationNode()
__C.model.solver.scheduler.cos_anneal.first_cycle_steps = 1841 * 5
__C.model.solver.scheduler.cos_anneal.cycle_mult = -1
__C.model.solver.scheduler.cos_anneal.max_lr = 1e-4
__C.model.solver.scheduler.cos_anneal.min_lr = 1e-7
__C.model.solver.scheduler.cos_anneal.warmup_steps = 1841
__C.model.solver.scheduler.cos_anneal.gamma = 0.9 # next cycle max_lr = 0.9 * previous_max_lr
__C.model.solver.scheduler.cos_anneal.last_epoch = -1

# epochs
__C.model.solver.total_epochs = 40
__C.model.solver.amp = False

__C.model.solver.loss = ConfigurationNode()
__C.model.solver.loss.name = 'cross_entropy'




def get_cfg_defaults():
    """
    Get a yacs cfgNode object with default values
    """
    return __C.clone()


def combine_cfgs(path_cfg_data: Path = None, path_cfg_override: Path = None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override = Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4 respectively
    # .env > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 4:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 3:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 2:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    return cfg_base
