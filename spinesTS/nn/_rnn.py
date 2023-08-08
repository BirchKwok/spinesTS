from typing import Any

import torch
from torch import nn
from spinesTS.layers import DifferentialLayer
from spinesTS.base import TorchModelMixin


class StackingRNNCell(nn.Module):
    def __init__(self, in_features, out_features, num_layers=2, bias=True,
                 dropout=0., bidirectional=False, diff_n=1):
        super(StackingRNNCell, self).__init__()

        self.in_features, self.out_features = in_features, out_features

        if not bidirectional:
            decoder_input_features = out_features
        else:
            decoder_input_features = out_features * 2

        self.differential_layer = DifferentialLayer(axis=-1, diff_n=diff_n)
        self.encoder = nn.LSTM(in_features-diff_n, out_features, num_layers=num_layers,
                               bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=False)
        self.decoder = nn.LSTM(decoder_input_features, out_features, num_layers=num_layers,
                               bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=False)

        self.linear = nn.Linear(decoder_input_features, out_features)

    def forward(self, x):
        x = self.differential_layer(x)
        if x.ndim == 2:
            torch.unsqueeze(x, dim=0)

        output, (h0, c0) = self.encoder(x)
        output, (hn1, cn1) = self.decoder(output, (h0, c0))

        return self.linear(output.squeeze())


class StackingRNN(TorchModelMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_layers=2,
                 loss_fn='mse',
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 diff_n=1,
                 learning_rate: float = 0.01,
                 random_seed: int = 42,
                 device=None
                 ) -> None:
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(num_layers=num_layers, bias=bias,
                                                             dropout=dropout,
                                                             bidirectional=bidirectional, diff_n=diff_n)

    def call(self, num_layers, bias,
             dropout,
             bidirectional, diff_n) -> tuple:
        model = StackingRNNCell(
            in_features=self.in_features,
            out_features=self.out_features,
            num_layers=num_layers, bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            diff_n=diff_n
        )
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(self,
            X_train: Any,
            y_train: Any,
            epochs: int = 1000,
            batch_size: str = 'auto',
            eval_set: Any = None,
            monitor: str = 'val_loss',
            min_delta: int = 0,
            patience: int = 10,
            lr_scheduler: str = 'ReduceLROnPlateau',
            lr_scheduler_patience: int = 10,
            lr_factor: float = 0.7,
            restore_best_weights: bool = True,
            verbose: bool = True,
            **kwargs: Any) -> Any:
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
