from torch import nn
from yacs.config import CfgNode
from ..layers.linear import LinearLayer
from ..builder.head_builder import HEAD_REGISTRY


@HEAD_REGISTRY.register('simple_classification_head')
def build_simple_classification_head(head_cfg: CfgNode) -> nn.Module:
    return SimpleClassificationHead(head_cfg)


class SimpleClassificationHead(nn.Module):
    def __init__(self, head_cfg: CfgNode):
        super(SimpleClassificationHead, self).__init__()
        self.fc_layers = []
        input_dim = head_cfg.input_dims
        for hidden_dim in head_cfg.hidden_dims:
            self.fc_layers.append(
                LinearLayer(input_dim, hidden_dim, activation=head_cfg.activation,
                            bn=head_cfg.batch_norm, dropout_rate=head_cfg.dropout)
            )
            input_dim = hidden_dim
        output_dims = head_cfg.output_dims
        self.fc_layers.append(
            LinearLayer(input_dim, output_dims, activation=None, bn=False, dropout_rate=-1)
        )
        self.fc_layers = nn.Sequential(*self.fc_layers)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self,x):
        return self.fc_layers(x)


