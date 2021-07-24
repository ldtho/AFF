import tarfile
import os
import sys
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx
import fire
from datetime import datetime
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm_notebook, tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
from pytorch_toolbelt.inference import tta
from torch.cuda.amp import GradScaler, autocast
from config.config import get_cfg_defaults, combine_cfgs
from pathlib import Path
from src.dataset.MURADataset import build_dataloader
from torch.utils.tensorboard import SummaryWriter
from src.config.config import get_cfg_defaults
from src.model.builder.model_builder import build_model
from src.model.builder.backbone_builder import build_backbone
from src.model.builder.baseline_arch import build_baseline_model
from src.model.builder.optimizer_builder import build_optimizer
from src.model.builder.scheduler_builder import build_scheduler


def train(config_path, data_path, output_path):
    cfg = get_cfg_defaults()
    if config_path is not None:
        cfg = combine_cfgs(Path(config_path))
    cfg.output_path = output_path
    train_data_path = cfg.dataset.train_data_path
    val_data_path = cfg.dataset.val_data_path

    timestamp = datetime.now().isoformat(sep="T", timespec='auto')
    name_timestamp = timestamp.replace(":", "_")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    backup_dir = os.path.join(output_path, "model_backups")
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)

    results_dir = os.path.join(output_path, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # initializa tensorboard
    writer_tensorboard = SummaryWriter(log_dir=results_dir + "logs_tensorflow")
    # save config
    cfg.dump(stream=open(os.path.join(results_dir, f'config_{name_timestamp}.yaml'), 'w'))
    # file path to store state of the models
    state_fpath = os.path.join(output_path, 'model.pt')

    # performance path
    parf_path = os.path.join(results_dir, 'trace.p')
    perf_trace = []

    # Data loader
    train_loader = build_dataloader(cfg.dataset, True)
    valid_loader = build_dataloader(cfg.dataset, False)
    # Update the config for later scheduler build
    cfg.model.solver.scheduler.steps_per_epoch = len(train_loader)

    model = build_model(cfg.model)

    solver_cfg = cfg.model.solver

    loss_fn = solver_cfg.loss.name

    current_epoch = 0

    multi_gpu_training = cfg.multi_gpu_training

    if cfg.resume_path != '':
        checkpoint = torch.load(cfg.resume_path, map_location='cpu')
        current_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
    if multi_gpu_training:
        model = torch.nn.DataParallel(model)
    device = torch.device(cfg.device)
    model = model.to(device)

    # Optimizer, scheduler, amp
    optim_cfg = solver_cfg.optimizer
    optimizer = build_optimizer(optim_cfg)

    scheduler_cfg = solver_cfg.scheduler
    scheduler_type = scheduler_cfg.name
    scheduler = build_scheduler(optimizer, scheduler_cfg=scheduler_cfg)

    if cfg.resume_path != '':
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])

    ## TO-DO: Continue the training loop


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    model_cfg = cfg.model
    model = build_model(model_cfg)
    x = torch.rand((2, 3, 320, 320))
    print(model)
    # fire.Fire()
