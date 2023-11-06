import torch
from torch import nn

from spinesTS.layers import Time2Vec
from spinesTS.base import TorchModelMixin, ForecastingMixin
from spinesTS.nn.utils import get_weight_norm

# in case of using MPS
import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


class T2V(nn.Module):
    def __init__(self, in_features, out_features, device=None):
        super(T2V, self).__init__()
        weight_norm = get_weight_norm(device)

        self.in_features, self.out_features = in_features, out_features
        self.t2v = Time2Vec(self.in_features, in_features)

        self.lstm = nn.LSTM(self.in_features * 2 - 1, in_features,
                            batch_first=True, bidirectional=False, bias=False)

        self.linear = weight_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        x = self.t2v(x)
        output, (h, c) = self.lstm(x.unsqueeze(1))

        return self.linear(output.squeeze())


class Time2VecNet(TorchModelMixin, ForecastingMixin):
    def __init__(self, in_features, out_features, learning_rate=0.001,
                 random_seed=42, device='auto', loss_fn='mae'):
        super(Time2VecNet, self).__init__(random_seed, device=device, loss_fn=loss_fn)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = self.call(device=device)
        self.loss_fn_name = loss_fn

    def call(self, device):
        model = T2V(self.in_features, self.out_features, device=device)
        loss_fn = self.loss_fn
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(
            self,
            X_train,
            y_train,
            epochs=1000,
            batch_size='auto',
            eval_set=None,
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            verbose=True,
            loss_type='min',
            **kwargs
    ):
        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type=loss_type,
                           metrics_name=self.loss_fn_name,
                           monitor=monitor, lr_scheduler=lr_scheduler,
                           lr_scheduler_patience=lr_scheduler_patience,
                           lr_factor=lr_factor,
                           min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                           verbose=verbose, **kwargs)
