from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.layers import PositionalEncoding


class EncoderDecoderBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 dropout=0.):
        super(EncoderDecoderBlock, self).__init__()

        self.encoder = nn.LSTM(in_features, in_features, num_layers=1,
                               bias=bias, dropout=dropout,
                               bidirectional=False, batch_first=True)

        self.decoder_output = nn.Linear(in_features, in_features)

        self.position_encoder = PositionalEncoding(in_features)

        self.linear_1 = nn.Linear(in_features, 1024)
        self.linear_2 = nn.Linear(1024, out_features)
        self.selu = nn.SELU()

    def forward(self, x):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=1)

        _, (h, _) = self.encoder(x)

        x = self.position_encoder(x)
        x = x.squeeze() + h.squeeze()

        output = self.decoder_output(x.squeeze())

        output = self.selu(self.linear_1(output.squeeze()))

        output = self.linear_2(output)
        return output


class Seq2SeqBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 dropout=0.,):
        super(Seq2SeqBlock, self).__init__()

        self.in_features, self.out_features = in_features, out_features

        self.forward_block = EncoderDecoderBlock(
                    in_features=in_features,
                    out_features=in_features,
                    bias=bias, dropout=dropout
        )

        self.backward_block = EncoderDecoderBlock(
            in_features=in_features,
            out_features=in_features,
            bias=bias, dropout=dropout
        )

        self.linear_0 = nn.Linear(in_features * 2, 1024)
        self.selu = nn.SELU()
        self.linear_1 = nn.Linear(1024, out_features)

        self.position_encoder1 = PositionalEncoding(in_features, add_x=False)
        self.position_encoder2 = PositionalEncoding(in_features, add_x=False)

    def forward(self, x):
        backward_output = self.backward_block(torch.flip(x, dims=[-1]))
        forward_output = self.forward_block(x)

        if backward_output.ndim == 1:
            backward_output = backward_output.unsqueeze(0)
        if forward_output.ndim == 1:
            forward_output = forward_output.unsqueeze(0)

        forward_output = forward_output + self.position_encoder1(forward_output).squeeze().mean(dim=1)
        backward_output = backward_output + self.position_encoder2(backward_output).squeeze().mean(dim=1)

        output = torch.concat((forward_output, backward_output), dim=-1)
        output = self.linear_0(output.squeeze())
        output = self.selu(output)

        return self.linear_1(output)


class StackingRNN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 loss_fn='mae',
                 bias=True,
                 dropout=0.2,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='cpu'
                 ) -> None:
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(bias=bias,
                                                             dropout=dropout)

    def call(self, bias,
             dropout) -> tuple:
        model = Seq2SeqBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            dropout=dropout,
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
            lr_scheduler: Union[str, None] = 'ReduceLROnPlateau',
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
