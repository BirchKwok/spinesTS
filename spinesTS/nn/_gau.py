from typing import Any, Union

import torch
from torch import nn

from spinesTS.layers import GAU
from spinesTS.base import TorchModelMixin, ForecastingMixin


class SkipConnect(nn.Module):
    def __init__(self, in_features, level=2, **kwargs):
        super(SkipConnect, self).__init__()
        self.gau = nn.ModuleList([
            GAU(in_features, **kwargs)
            for i in range(level)
            ])
        self.level = level

    def forward(self, x):
        x = self.gau[0](x)
        if self.level > 1:
            x_ = x.clone().detach()
            for i in range(1, self.level):
                x = self.gau[i](x)
                if i % 2 == 0 and i != (self.level - 1):
                    x = x + x_
                    x_ = x.clone().detach()
        return x


class GAUBase(nn.Module):
    def __init__(self, in_shapes, out_features, flip_features=False, level=2, skip_connect=True, dropout=0.):
        super(GAUBase, self).__init__()
        self.in_shapes_type = type(in_shapes)

        self.in_features, self.out_features = \
            in_shapes[-1] if self.in_shapes_type == tuple else in_shapes, out_features

        self.gau = SkipConnect(self.in_features, level, skip_connect=skip_connect, dropout=dropout)

        if flip_features:
            self.gau2 = SkipConnect(self.in_features, level, skip_connect=skip_connect, dropout=dropout)

            ln_layer_in_fea = \
                2 * in_shapes[0] * in_shapes[1] if self.in_shapes_type == tuple else 2 * self.in_features
        else:
            ln_layer_in_fea = in_shapes[0] * in_shapes[1] if self.in_shapes_type == tuple else self.in_features

        self.flip_features = flip_features
        self.linear = nn.Linear(ln_layer_in_fea, out_features)

    def forward(self, x):
        x1 = self.gau(x)
        if self.flip_features:
            _x = x.clone()
            x2 = self.gau2(torch.flip(_x, dims=[1]))
            x = torch.concat((x1, x2), dim=-1)
        else:
            x = x1

        if x.ndim == 2:
            return self.linear(x)

        return self.linear(x.reshape(-1, x.shape[1] * x.shape[2]))


class GAUNet(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 flip_features: bool = False,
                 level: int = 2,
                 skip_connect: bool = True,
                 dropout: float = 0.,
                 learning_rate: float = 0.01,
                 random_seed: int = 42,
                 device=None,
                 loss_fn='mae'
                 ) -> None:
        super(GAUNet, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.flip_features = flip_features
        self.model, self.loss_fn, self.optimizer = self.call(level, skip_connect=skip_connect, dropout=dropout)

    def call(self,
             level: int = 2,
             **kwargs: Any) -> tuple:
        model = GAUBase(self.in_features, self.out_features,
                        flip_features=self.flip_features, level=level, **kwargs)
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
