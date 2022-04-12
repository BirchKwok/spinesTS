import torch
from torch import nn
from spinesTS.layers import Time2Vec
from spinesTS.base import TorchModelMixin
from spinesTS.utils import seed_everything


class T2V(nn.Module):
    def __init__(self, in_features, out_features, flip_features=False):
        super(T2V, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.t2v = Time2Vec(in_features)
        if flip_features:
            self.t2v2 = Time2Vec(in_features)
            ln_layer_in_fea = 4 * in_features
        else:
            ln_layer_in_fea = 2 * in_features

        self.linear = nn.Linear(ln_layer_in_fea, out_features)
        self.flip_features = flip_features

    def forward(self, x):
        x1 = self.t2v(x)
        if self.flip_features:
            _x = x.clone()
            x2 = self.t2v2(torch.flip(_x, dims=[1]))
            x = torch.concat((x1, x2), dim=-1)
        else:
            x = x1

        return self.linear(x)


class Time2VecNet(TorchModelMixin):
    def __init__(self, in_features, out_features, flip_features=False, learning_rate=0.01, random_seed=42):
        seed_everything(random_seed)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.flip_features = flip_features
        self.model, self.loss_fn, self.optimizer = self.call()

    def call(self):
        model = T2V(self.in_features, self.out_features, flip_features=self.flip_features)
        loss_fn = nn.HuberLoss()
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
            use_lr_scheduler=False,
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            verbose=True
    ):
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        return self._fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                         monitor=monitor, use_lr_scheduler=use_lr_scheduler,
                         lr_scheduler_patience=lr_scheduler_patience,
                         lr_factor=lr_factor,
                         min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                         verbose=verbose)

    def predict(self, x):
        assert self.model is not None, "model not fitted yet."
        return self._predict(x)





















