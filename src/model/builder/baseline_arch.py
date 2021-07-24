import torch
from src.model.builder.backbone_builder import build_backbone
from src.model.builder.head_builder import build_head
from torch import nn
from yacs.config import CfgNode

from src.model.builder.model_builder import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register('baseline')
def build_baseline_model(model_cfg: CfgNode) -> nn.Module:
    return BaselineModel(model_cfg)


class BaselineModel(nn.Module):
    def __init__(self, model_cfg: CfgNode):
        super(BaselineModel, self).__init__()
        self.backbone = build_backbone(model_cfg.backbone)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.head = build_head(model_cfg.head)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.head(x)
        return x

