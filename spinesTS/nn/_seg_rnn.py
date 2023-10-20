from typing import Any, Union

import torch
from torch import nn

from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.layers import PositionalEncoding3D


def get_even_blocks(in_features, window_size):
    """
    将输入数据分割成多个块，每个块的大小为window_size
    :param in_features:
    :return:
    """

    blocks = in_features // window_size + int(in_features % window_size != 0)

    while (blocks % 2 != 0 or window_size % 2 != 0) and window_size >= 6:
        window_size -= 1
        blocks = in_features // window_size + int(in_features % window_size != 0)

    return blocks, window_size


class SegmentationBlock(nn.Module):
    """将输入数据分割成多个块，每个块的大小为window_size"""

    def __init__(self, in_features, window_size=12):
        super(SegmentationBlock, self).__init__()

        self.blocks, self.window_size = get_even_blocks(in_features, window_size)

        if self.blocks % 2 != 0:
            self.blocks += 1

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.window_size, self.window_size),
                nn.ReLU()
            )
            for i in range(self.blocks)
        ])

    def forward(self, x):
        assert x.ndim == 2

        if self.blocks == 1:
            xs = [x]
        else:
            xs = list(torch.split(x, self.window_size, dim=1))
            xs[-1] = torch.concat([xs[-1], torch.zeros(xs[-1].shape[0],
                                                       self.window_size - xs[-1].shape[1])], dim=1)

        res = []
        for i in range(self.blocks):
            res.append(self.encoders[i](xs[i]))

        return torch.stack(res, 1)  # [batch_size, blocks, window_size]


class SegRNNBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(SegRNNBlock, self).__init__()

        if in_features < 24:
            raise ValueError("The number of input features must be equal or greater than 24.")

        self.splitter = SegmentationBlock(in_features=in_features, window_size=12)
        window_size = self.splitter.window_size

        self.rnn = nn.GRU(window_size, window_size, num_layers=1,
                          bias=True, bidirectional=False, batch_first=True)

        self.block_position_encoder = PositionalEncoding3D(d_model=window_size)
        self.channel_position_encoder = PositionalEncoding3D(d_model=self.splitter.blocks)

        out_window_size = out_features // self.splitter.blocks + int(out_features % self.splitter.blocks > 0)

        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(window_size, 1024),
                nn.Linear(1024, out_window_size)
            )
            for i in range(self.splitter.blocks)
        ])

        self.out_features = out_features

    def forward(self, x):
        x = self.splitter(x)  # [batch_size, blocks, window_size]

        _, h = self.rnn(x)

        pos_enc = self.block_position_encoder(x)

        channel_encs = self.channel_position_encoder(torch.randn(x.shape[0], x.shape[1], self.splitter.blocks))

        channel_enc = torch.empty(pos_enc.size(0), pos_enc.size(1), pos_enc.size(2), device=x.device)

        # 沿轴复制元素
        for _ in range(channel_encs.shape[2]):
            for _2 in range(self.splitter.blocks):
                if _ * self.splitter.blocks + _2 < pos_enc.size(2):
                    channel_enc[:, :, _ * self.splitter.blocks + _2] = channel_encs[:, :, _]

        x_enc = x + pos_enc + channel_enc

        output, _ = self.rnn(x_enc, h)

        res = []
        for i in range(self.splitter.blocks):
            res.append(self.decoders[i](output[:, i, :]))

        res = torch.concat(res, dim=1)

        return res[:, :self.out_features]


class SegRNN(TorchModelMixin, ForecastingMixin):
    """长序列预测模型

    References
    ----------
    LIN S. SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting[J]. 2023.
    arXiv preprint arXiv:2308.11200
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 loss_fn='mae',
                 dropout=0.1,
                 learning_rate: float = 0.001,
                 random_seed: int = 42,
                 device='cpu'
                 ) -> None:
        super(SegRNN, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(dropout=dropout)

    def call(self, dropout) -> tuple:
        model = SegRNNBlock(
            in_features=self.in_features,
            out_features=self.out_features,
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
