# pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
from src.model.solver.scheduler import CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, MultiStepLR
from torch.optim.optimizer import Optimizer
from yacs.config import CfgNode


def build_scheduler(optimizer: Optimizer, scheduler_cfg: CfgNode):
    scheduler_type = scheduler_cfg.name
    if scheduler_type == 'unchange':
        return None
    elif scheduler_type == 'multi_steps':
        scheduler = MultiStepLR(optimizer,
                                milestones=scheduler_cfg.multi_steps_lr_milestones,
                                gamma=scheduler_cfg.lr_reduce_gamma,
                                last_epoch=-1)
        return scheduler
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      patience=scheduler_cfg.patience,
                                      factor=scheduler_cfg.lr_reduce_gamma)
        return scheduler
    elif scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(optimizer,
                               max_lr=scheduler_cfg.one_cycle.max_lr,
                               steps_per_epoch=scheduler_cfg.one_cycle.steps_per_epoch,
                               epochs=scheduler_cfg.one_cycle.total_epochs,
                               pct_start=scheduler_cfg.one_cycle.pct_start,
                               anneal_strategy=scheduler_cfg.one_cycle.anneal_strategy,
                               div_factor=scheduler_cfg.one_cycle.div_factor,
                               cycle_momentum=scheduler_cfg.one_cycle.cycle_momentum,
                               )
        return scheduler
    elif scheduler_type == "cos_anneal_warmup_restart":
        # optimizer (Optimizer): Wrapped optimizer.
        # first_cycle_steps (int): First cycle step size.
        # cycle_mult(float): Cycle steps magnification. Default: -1.
        # max_lr(float): First cycle's max learning rate. Default: 0.1.
        # min_lr(float): Min learning rate. Default: 0.001.
        # warmup_steps(int): Linear warmup step size. Default: 0.
        # gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        # last_epoch (int): The index of last epoch. Default: -1.
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=scheduler_cfg.cos_anneal.first_cycle_steps,
                                                  cycle_mult=scheduler_cfg.cos_anneal.cycle_mult,
                                                  max_lr=scheduler_cfg.cos_anneal.max_lr,
                                                  min_lr=scheduler_cfg.cos_anneal.min_lr,
                                                  warmup_steps=scheduler_cfg.cos_anneal.warmup_steps,
                                                  gamma=scheduler_cfg.cos_anneal.gamma,
                                                  last_epoch=scheduler_cfg.cos_anneal.last_epoch,
                                                  )
        return scheduler

    else:
        raise Exception('Invalid scheduler name, options: \n'
                        'unchange/multi_steps/reduce_on_plateau/one_cycle/cos_anneal_warmup_restart')
