import torch
from torch import nn

from spinesTS.utils import seed_everything
from spinesTS.base import TorchModelMixin
from spinesTS.layers import ResDenseBlock, Hierarchical1d


class WeightedEncoder(nn.Module):
    def __init__(self, in_features, rin=True):
        super(WeightedEncoder, self).__init__()
        self.hierarchical1d = Hierarchical1d()

        self.RIN = rin

        self.odd_shape = in_features // 2 + in_features % 2
        self.even_shape = in_features // 2
        self.res_dense_blocks1 = ResDenseBlock(self.odd_shape, kernel_size=5)

        self.res_dense_blocks2 = ResDenseBlock(self.even_shape, kernel_size=5)

        self.res_dense_blocks3 = ResDenseBlock(self.even_shape, kernel_size=3)

        self.res_dense_blocks4 = ResDenseBlock(self.even_shape, kernel_size=3)

        self.padding = nn.ReflectionPad1d((0, 1))

        self.lstms = nn.ModuleList([
            nn.LSTM(self.odd_shape, self.odd_shape), nn.LSTM(self.even_shape, self.even_shape)
        ])

        if self.RIN:
            self.affine_weight_odd = nn.Parameter(torch.ones(1, self.odd_shape))
            self.affine_bias_odd = nn.Parameter(torch.zeros(1, self.odd_shape))
            self.affine_weight_even = nn.Parameter(torch.ones(1, self.even_shape))
            self.affine_bias_even = nn.Parameter(torch.zeros(1, self.even_shape))

    def rin_transform(self, x, name='odd'):
        means = x.mean(1, keepdim=True).detach()
        # mean
        x = x - means
        # var
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        # affine
        # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
        x = x * (self.affine_weight_odd if name == 'odd' else self.affine_weight_even) + \
            (self.affine_bias_odd if name == 'odd' else self.affine_bias_even)

        return x, stdev, means

    def rin_inverse(self, x, stdev, means, name='odd'):
        x = x - (self.affine_bias_odd if name == 'odd' else self.affine_bias_even)
        x = x / ((self.affine_weight_odd if name == 'odd' else self.affine_weight_even) + 1e-10)
        x = x * stdev
        x = x + means
        return x

    def forward(self, x):
        x_odd, x_even = self.hierarchical1d(x)
        if self.RIN:
            x_odd, odd_stdev, odd_means = self.rin_transform(x_odd, name='odd')
            x_even, even_stdev, even_means = self.rin_transform(x_even, name='even')

        if x_odd.shape[1] > x_even.shape[1]:
            c = self.res_dense_blocks2(x_even, x_odd[:, 1:])
            d = self.res_dense_blocks1(x_odd, self.padding(x_even))

            x_odd_update = d  # - self.padding(self.res_dense_blocks3(c, x_even))
            x_even_update = c  # + self.res_dense_blocks4(d[:, 1:], x_odd[:, 1:])

        else:
            c = self.res_dense_blocks2(x_even, x_odd)
            d = self.res_dense_blocks1(x_odd, x_even)

            x_odd_update = d  # - self.res_dense_blocks3(c, x_even)
            x_even_update = c  # + self.res_dense_blocks4(d, x_odd)

        x_odd_update = torch.squeeze(self.lstms[0](x_odd_update.view(1, -1, self.odd_shape))[0])
        x_even_update = torch.squeeze(self.lstms[1](x_even_update.view(1, -1, self.even_shape))[0])

        if self.RIN:
            x_odd_update = self.rin_inverse(x_odd_update, odd_stdev, odd_means, name='odd')
            x_even_update = self.rin_inverse(x_even_update, even_stdev, even_means, name='even')

        return x_odd_update, x_even_update


class EncoderTree(nn.Module):
    def __init__(self, in_features, level):
        super(EncoderTree, self).__init__()
        self.level = level - 1
        self.encoder = WeightedEncoder(in_features)
        if self.level != 0:
            odd_shape = in_features // 2 + in_features % 2
            even_shape = in_features // 2
            self.sub_odd_encoder = WeightedEncoder(odd_shape)
            self.sub_even_encoder = WeightedEncoder(even_shape)

    def forward(self, x):
        if self.level == 0:
            return torch.cat(self.encoder(x), dim=-1)
        else:
            x_odd, x_even = self.encoder(x)
            return torch.cat((torch.cat(self.sub_odd_encoder(x_odd), -1), torch.cat(self.sub_even_encoder(x_even), -1)),
                             dim=-1)


class WeightedDenseRNNBase(nn.Module):
    def __init__(self, in_features, output_features, level=4):
        super(WeightedDenseRNNBase, self).__init__()
        self.in_features, self.output_features = in_features, output_features
        self.encoder_tree = EncoderTree(in_features, level=level)

        self.output_layer = nn.Linear(
            in_features,
            self.output_features
        )

    def forward(self, x):
        x = self.encoder_tree(x)

        return self.output_layer(x)


class WeightedDenseRNN(TorchModelMixin):
    """
    spinesTS MLP pytorch-model

    """

    def __init__(
            self,
            in_features,
            out_features,
            learning_rate=0.0001,
            level=1,
            random_seed=42
    ):
        assert in_features > 1, "in_features must be greater than 1."
        seed_everything(random_seed)
        self.out_features, self.in_features = out_features, in_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = None, None, None

        self.level = level

        self.model, self.loss_fn, self.optimizer = self.call(self.in_features)

    def call(self, in_features):
        model = WeightedDenseRNNBase(in_features, self.out_features, self.level)
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
        """
        lr_Scheduler: torch.optim.lr_scheduler class,
            only support to ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
        """
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        return self._fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                         monitor=monitor, lr_scheduler=lr_scheduler,
                         lr_scheduler_patience=lr_scheduler_patience,
                         lr_factor=lr_factor,
                         min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                         verbose=verbose, **kwargs)

    def predict(self, x):
        assert self.model is not None, "model not fitted yet."
        return self._predict(x)
