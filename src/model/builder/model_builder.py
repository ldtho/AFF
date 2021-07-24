from src.tools.registry import Registry
from torch import nn
from yacs.config import CfgNode

META_ARCH_REGISTRY = Registry()


def build_model(model_cfg: CfgNode) -> nn.Module:
    meta_arch = model_cfg.meta_architecture
    model = META_ARCH_REGISTRY.get(meta_arch)(model_cfg)
    return model


