from src.tools.registry import Registry
from torchvision import models
from torch import nn
from yacs.config import CfgNode
BACKBONE_REGISTRY = Registry()

def build_backbone(backbone_cfg: CfgNode) -> nn.Module:
    """
    build backbone
    :param backbone_cfg
    :return: backbone network
    """
    backbone = BACKBONE_REGISTRY[backbone_cfg.name](backbone_cfg)
    return backbone