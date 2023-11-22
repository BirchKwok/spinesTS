from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.layers import LearnablePositionalEncoding


class EncoderDecoderBlock(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            dropout=0.1
    ):
        super(EncoderDecoderBlock, self).__init__()

        self.encoder = nn.LSTM(in_features, in_features, num_layers=1,
                               bias=bias,
                               bidirectional=False, batch_first=True)

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.position_encoder = LearnablePositionalEncoding(in_features)

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(x.size(0), 1, x.size(1))
        elif x.ndim == 1:
            x = x.view(1, 1, x.size(0))

        _, (h, _) = self.encoder(x)
        x = self.position_encoder(x)

        x = x.squeeze() + h.squeeze()

        if x.ndim == 1:
            x = x.view(1, -1)

        return self.decoder(x.view(x.size(0), -1))


class Seq2SeqBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dropout=0.1, blocks=2):
        super(Seq2SeqBlock, self).__init__()

        self.in_features, self.out_features = in_features, out_features

        self.enc_dnc = nn.Sequential(
            *nn.ModuleList([
                EncoderDecoderBlock(
                    in_features=in_features,
                    out_features=in_features,
                    bias=bias,
                    dropout=dropout
                ) for _ in range(blocks)]
            )
        )

        self.position_encoder = LearnablePositionalEncoding(in_features)

        self.output_layer = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(x.size(0), 1, x.size(1))

        output = self.enc_dnc(x)

        # 残差连接
        output = self.position_encoder(output.unsqueeze(1)).squeeze() + x.squeeze()

        return self.output_layer(output.squeeze())


class StackingRNN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout=0.1,
                 blocks=2,
                 loss_fn='mae',
                 bias=False,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto'
                 ) -> None:
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(bias=bias, dropout=dropout, blocks=blocks)
        self.loss_fn_name = loss_fn

    def call(self, bias, dropout, blocks) -> tuple:
        model = Seq2SeqBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            dropout=dropout,
            blocks=blocks
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
