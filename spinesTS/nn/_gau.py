from typing import Any, Union

import torch
from torch import nn

from spinesUtils.preprocessing import unsqueeze_if

from spinesTS.layers import GAU, PositionalEncoding
from spinesTS.base import TorchModelMixin, ForecastingMixin


class GAUBlock(nn.Module):
    def __init__(self, in_features, level=2, **kwargs):
        super(GAUBlock, self).__init__()
        self.gau = nn.ModuleList([
            nn.Sequential(
                PositionalEncoding(in_features, add_x=True),
                GAU(in_features, **kwargs)
            )
            for i in range(level)
            ])
        self.level = level

    def forward(self, x):
        x = unsqueeze_if(x, x.ndim == 2, 1)
        for i in self.gau:
            x = i(x)

        return x


class GAUBase(nn.Module):
    def __init__(self, in_shapes, out_features, level=2):
        super(GAUBase, self).__init__()
        self.in_shapes_type = type(in_shapes)

        self.in_features, self.out_features = \
            in_shapes[-1] if self.in_shapes_type == tuple else in_shapes, out_features

        self.gau = GAUBlock(self.in_features, level)

        ln_layer_in_fea = in_shapes[0] * in_shapes[1] if self.in_shapes_type == tuple else self.in_features

        self.linear = nn.Linear(ln_layer_in_fea, out_features)

    def forward(self, x):
        x = self.gau(x)

        if x.ndim == 2:
            return self.linear(x)

        return self.linear(x.reshape(-1, x.shape[1] * x.shape[2]))


class GAUNet(TorchModelMixin, ForecastingMixin):
    def __init__(self,
                 in_features: Any,
                 out_features: Any,
                 level: int = 2,
                 learning_rate: float = 0.01,
                 random_seed: int = 42,
                 device='auto',
                 loss_fn='mae'
                 ) -> None:
        super(GAUNet, self).__init__(random_seed, device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.model, self.loss_fn, self.optimizer = self.call(level)

    def call(self,
             level: int = 2) -> tuple:
        model = GAUBase(self.in_features, self.out_features,
                        level=level)
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
