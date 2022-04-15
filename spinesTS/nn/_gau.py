import torch
from torch import nn
from spinesTS.layers import GAU
from spinesTS.base import TorchModelMixin


class GAUBase(nn.Module):
    def __init__(self, in_features, out_features, flip_features=False):
        super(GAUBase, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.gau = GAU(in_features)
        if flip_features:
            self.gau2 = GAU(in_features)
            ln_layer_in_fea = 2 * in_features
        else:
            ln_layer_in_fea = in_features

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

        return self.linear(x)


class GAUNet(TorchModelMixin):
    def __init__(self, in_features, out_features, flip_features=False, learning_rate=0.01, random_seed=42):
        super(GAUNet, self).__init__(random_seed)
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.flip_features = flip_features
        self.model, self.loss_fn, self.optimizer = self.call()

    def call(self):
        model = GAUBase(self.in_features, self.out_features, flip_features=self.flip_features)
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
            lr_scheduler='ReduceLROnPlateau',
            lr_scheduler_patience=10,
            lr_factor=0.7,
            restore_best_weights=True,
            verbose=True,
            **kwargs
    ):
       
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        return super().fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                         monitor=monitor, lr_scheduler=lr_scheduler,
                         lr_scheduler_patience=lr_scheduler_patience,
                         lr_factor=lr_factor,
                         min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                         verbose=verbose, **kwargs)
