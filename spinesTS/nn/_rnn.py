from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.layers import PositionalEncoding


class EncoderDecoderBlock(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False
    ):
        super(EncoderDecoderBlock, self).__init__()

        self.encoder = nn.LSTM(in_features, in_features, num_layers=1,
                               bias=bias,
                               bidirectional=False, batch_first=True)

        self.decoder_output = nn.Linear(in_features, in_features)

        self.position_encoder = PositionalEncoding(in_features)

        self.linear = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=1)

        _, (h, _) = self.encoder(x)

        x = self.position_encoder(x)
        x = x.squeeze() + h.squeeze()

        output = self.decoder_output(x.squeeze())

        output = self.gelu(self.linear(output.squeeze()))

        return output


class Seq2SeqBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Seq2SeqBlock, self).__init__()

        self.in_features, self.out_features = in_features, out_features

        self.forward_block = EncoderDecoderBlock(
            in_features=in_features,
            out_features=in_features,
            bias=bias
        )

        self.linear_0 = nn.Linear(in_features, 256)
        self.gelu = nn.GELU()
        self.linear_1 = nn.Linear(256, out_features)

        self.position_encoder1 = PositionalEncoding(in_features, add_x=False)

    def forward(self, x):
        forward_output = self.forward_block(x)

        if forward_output.ndim == 1:
            forward_output = forward_output.unsqueeze(0)

        output = forward_output + self.position_encoder1(forward_output).squeeze().mean(dim=1)

        output = self.linear_0(output.squeeze())
        output = self.gelu(output)

        return self.linear_1(output)


class StackingRNN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 loss_fn='mae',
                 bias=False,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='auto'
                 ) -> None:
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(bias=bias)
        self.loss_fn_name = loss_fn

    def call(self, bias) -> tuple:
        model = Seq2SeqBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias
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
