# reference: https://github.com/locuslab/TCN/tree/master
from typing import Any, Union

import torch
import torch.nn as nn
from spinesUtils.preprocessing import unsqueeze_if

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.nn.utils import get_weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, device=None):
        super(TemporalBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=2, dropout=0.2, device=None):
        super(TemporalConvNet, self).__init__()
        layers = []

        dilation_size = 2
        in_channels = in_features
        out_channels = out_features
        layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                 padding=(kernel_size-1) * dilation_size, dropout=dropout, device=device)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = unsqueeze_if(x, x.ndim == 2, -1)
        x = self.network(x)

        return x.squeeze(-1)


class TCN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 kernel_size: int = 2,
                 dropout: float=0.2,
                 learning_rate: float = 0.01,
                 random_seed: int = 42,
                 device='auto',
                 loss_fn='mae'
                 ) -> None:
        super(TCN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.model, self.loss_fn, self.optimizer = self.call(kernel_size=kernel_size, dropout=dropout, device=device)

    def call(self,
             kernel_size: int = 2, dropout: float = 0.2, device='auto') -> tuple:
        model = TemporalConvNet(self.in_features, self.out_features, kernel_size=kernel_size, dropout=dropout,
                                device=device)
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(self,
            X_train: Any,
            y_train: Any,
            epochs: int = 1000,
            batch_size: Union[str, int] = 'auto',
            eval_set: Any = None,
            monitor: str = 'val_loss',
            min_delta: int = 0,
            patience: int = 10,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            verbose: bool = True,
            loss_type='min',
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)