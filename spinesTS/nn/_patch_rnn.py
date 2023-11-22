from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.layers._position_encoder import LearnablePositionalEncoding
from spinesTS.nn.utils import get_weight_norm


class SegmentationBlock(nn.Module):
    """将输入数据分割成多个块，每个块的大小为window_size"""

    def __init__(self, in_features, kernel_size=4, device=None):
        super(SegmentationBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.kernel_size = kernel_size

        self.encoder = nn.Sequential(
            LearnablePositionalEncoding(self.kernel_size),
            weight_norm(nn.Linear(self.kernel_size, self.kernel_size)),
            nn.GELU(),
            nn.BatchNorm1d(in_features - kernel_size + 1)
        )

    def forward(self, x):
        assert x.ndim == 2

        x = x.unfold(dimension=-1, size=self.kernel_size, step=1)

        return self.encoder(x)


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, device=None):
        super(FCBlock, self).__init__()
        weight_norm = get_weight_norm(device)

        self.fc_layer = nn.Sequential(
            weight_norm(nn.Linear(in_features, 512)),
            nn.GELU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(512, out_features))
        )

    def forward(self, x):
        return self.fc_layer(x)


class PatchRNNBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=4, dropout=0.1, device=None):
        super(PatchRNNBlock, self).__init__()
        self.splitter = SegmentationBlock(in_features, kernel_size, device=device)

        self.encoder_rnn = nn.LSTM(kernel_size, kernel_size, num_layers=1,
                                   bias=False, bidirectional=False, batch_first=True)

        self.decoder = FCBlock((kernel_size * (in_features - kernel_size + 1)), out_features, dropout=dropout,
                               device=device)

        self.batch_norm = nn.BatchNorm1d(in_features - kernel_size + 1)

    def forward(self, x):
        x = self.splitter(x)

        output, _ = self.encoder_rnn(x)
        output = self.batch_norm(output)
        output += x

        res = self.decoder(output[:, :, :].view(output.size(0), -1))

        return res


class PatchRNN(TorchModelMixin, ForecastingMixin):
    """长序列预测模型
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size=4,
                 dropout=0.1,
                 loss_fn='mae',
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto'
                 ) -> None:
        super(PatchRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(kernel_size, dropout)
        self.loss_fn_name = loss_fn

    def call(self, kernel_size, dropout) -> tuple:
        model = PatchRNNBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            device=self.device,
            kernel_size=kernel_size,
            dropout=dropout
        )
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
            patience: int = 100,
            lr_scheduler: Union[str, None] = 'CosineAnnealingLR',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            loss_type='min',
            verbose: bool = True,
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
