from torch import nn
import torch
from spinesTS.base import TorchModelMixin, ForecastingMixin


class Linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__()

    def forward(self, x):
        pass


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, level=4, bias=True):
        super(MLPBlock, self).__init__()
        self.blocks = nn.ModuleList(
            [

            ]
        )

    def forward(self, x):
        pass


class MLPModel(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 stack_num=2,
                 num_layers=2,
                 loss_fn='mae',
                 bias=True,
                 dropout=0.1,
                 bidirectional=True,
                 diff_n=1,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='cpu'
                 ) -> None:
        super(MLPModel, self).__init__(random_seed, device, loss_fn=loss_fn)

    def call(self, *args, **kwargs):
        pass
