# pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
import torch
from torch.optim.optimizer import Optimizer
from yacs.config import CfgNode


def build_optimizer(model: torch.nn.Module, optim_cfg: CfgNode) -> Optimizer:
    parameters = model.parameters()
    optim_type = optim_cfg.name
    lr = optim_cfg.base_lr
    weight_decay = optim_cfg.weight_decay
    if optim_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'SGD':
        sgd_cfg = optim_cfg.SGD
        momentum = sgd_cfg.momentum
        nesterov = sgd_cfg.nesterov
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov,
                                    weight_decay=weight_decay)
    elif optim_type == 'adamW':
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception('invalid optimizer, options: adam/adamW/SGD')
    return optimizer


