from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin


class EncoderDecoderBlock(nn.Module):
    def __init__(self, in_features, out_features, num_layers=1, bias=True,
                 dropout=0., bidirectional=False):
        super(EncoderDecoderBlock, self).__init__()

        if not bidirectional:
            decoder_input_features = in_features
        else:
            decoder_input_features = in_features * 2

        self.bidirectional = bidirectional

        self.encoder = nn.LSTM(in_features, in_features, num_layers=num_layers,
                               bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.decoder = nn.LSTM(decoder_input_features, in_features, num_layers=num_layers,
                               bias=bias, dropout=dropout, bidirectional=bidirectional, batch_first=True)

        self.linear = nn.Linear(decoder_input_features, out_features)
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(decoder_input_features, decoder_input_features)

    def forward(self, x, last_output):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=1)

        if self.bidirectional:
            x = torch.cat((x, x), dim=-1)

        if last_output.ndim == 2:
            last_output = torch.unsqueeze(last_output, dim=1)

        _, (h, c) = self.encoder(last_output)
        output, (_, _) = self.decoder(x, (h, c))

        attention_weights = torch.softmax(self.attention(output), dim=1)
        output = output * attention_weights  # Apply attention

        return self.selu(self.linear(output.squeeze()))


class Seq2SeqBlock(nn.Module):
    def __init__(self, in_features, out_features, stack_num=4, num_layers=1, bias=True,
                 dropout=0., bidirectional=False):
        super(Seq2SeqBlock, self).__init__()

        self.in_features, self.out_features = in_features, out_features

        self.blocks = nn.ModuleList(
            [
                EncoderDecoderBlock(
                    in_features=in_features,
                    out_features=in_features, num_layers=num_layers,
                    bias=bias, dropout=dropout, bidirectional=bidirectional
                ) for i in range(stack_num)
            ]
        )

        self.attention = nn.Linear(in_features, in_features)

        self.linear_0 = nn.Linear(in_features, 1024)
        self.selu = nn.SELU()

        self.linear_1 = nn.Linear(1024, out_features)

    def forward(self, x):
        last_output = x
        attention_weights = []
        for block in self.blocks:
            last_output = block(x, last_output)

            attention_weights.append(torch.softmax(self.attention(last_output), dim=1))

        attention_weights = torch.mean(torch.stack(attention_weights, dim=0), dim=0)
        last_output = last_output * attention_weights

        output = self.linear_0(last_output.squeeze())
        output = self.selu(output)

        return self.linear_1(output)


class StackingRNN(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 stack_num=10,
                 num_layers=1,
                 loss_fn='mae',
                 bias=True,
                 dropout=0.1,
                 bidirectional=True,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='cpu'
                 ) -> None:
        super(StackingRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(stack_num=stack_num, num_layers=num_layers, bias=bias,
                                                             dropout=dropout,
                                                             bidirectional=bidirectional)

    def call(self, stack_num, num_layers, bias,
             dropout,
             bidirectional) -> tuple:
        model = Seq2SeqBlock(
            in_features=self.in_features,
            out_features=self.out_features,
            stack_num=stack_num,
            num_layers=num_layers, bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
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
