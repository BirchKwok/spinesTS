import torch
from torch import nn

from spinesTS.utils import seed_everything
from spinesTS.base import TorchModelMixin
from spinesTS.layers import ResDenseBlock, Hierarchical1d


class WeightedEncoder(nn.Module):
    def __init__(self, in_features):
        super(WeightedEncoder, self).__init__()
        self.hierarchical1d = Hierarchical1d()

        odd_shape = in_features // 2 + in_features % 2
        even_shape = in_features // 2
        self.res_dense_blocks1 = ResDenseBlock(odd_shape)

        self.res_dense_blocks2 = ResDenseBlock(even_shape)

        self.res_dense_blocks3 = ResDenseBlock(odd_shape)

        self.res_dense_blocks4 = ResDenseBlock(even_shape)

        self.padding = nn.ReflectionPad1d((0, 1))

    def forward(self, x):
        x_odd, x_even = self.hierarchical1d(x)
        first_layer_1, first_layer_2 = x_odd.clone(), x_even.clone()

        if x_odd.shape[1] > x_even.shape[1]:
            c = x_odd[:, 1:].mul(torch.exp(self.res_dense_blocks2(x_even, first_layer_1)))
            d = self.padding(x_even).mul(torch.exp(self.res_dense_blocks1(x_odd, first_layer_1)))

            x_odd_update = d - self.padding(self.res_dense_blocks3(c))
            x_even_update = c + self.res_dense_blocks4(d[:, 1:])

        else:
            d = x_odd.mul(torch.exp(self.res_dense_blocks2(x_even, first_layer_1)))
            c = x_even.mul(torch.exp(self.res_dense_blocks1(x_odd, first_layer_1)))

            x_odd_update = d - self.res_dense_blocks3(c, c)
            x_even_update = c + self.res_dense_blocks4(d, d)

        return x_odd_update, x_even_update


class EncoderTree(nn.Module):
    def __init__(self, in_features, level):
        super(EncoderTree, self).__init__()
        self.level = level
        self.encoder = WeightedEncoder(in_features)
        if self.level != 0:
            pass


    def forward(self, x):
        return

class WeightedDenseRNNBase(nn.Module):
    def __init__(self, in_features, output_features, res_dense_blocks=4):
        super(WeightedDenseRNNBase, self).__init__()
        self.in_features, self.output_features = in_features, output_features

        self.input_layer_norm = nn.LayerNorm(self.in_features)
        self.encoder_hierarchical_layer = Hierarchical1d()

        linear_input_shape_odd = self.in_features // 2 + self.in_features % 2
        linear_input_shape_even = self.in_features // 2

        self.weighted_encoder = WeightedEncoder(linear_input_shape_odd, linear_input_shape_even)

        # self.encoder_hierarchical_lstm_layers = nn.ModuleList([
        #     nn.LSTM(linear_input_shape_odd, linear_input_shape_odd, 1, bidirectional=True),
        #     nn.LSTM(linear_input_shape_even, linear_input_shape_even, 1, bidirectional=True),
        # ])

        # self.decoder_hierarchical_lstm_layers = nn.ModuleList([
        #     nn.LSTM(linear_input_shape_odd * 2, linear_input_shape_odd, 1, bidirectional=True),
        #     nn.LSTM(linear_input_shape_even * 2, linear_input_shape_even, 1, bidirectional=True),
        # ])

        # self.encoder_module_list_1 = nn.ModuleList([
        #     ResDenseBlock(linear_input_shape_odd, linear_input_shape_odd)
        #     for i in range(res_dense_blocks)])
        #
        # self.encoder_module_list_2 = nn.ModuleList([
        #     ResDenseBlock(linear_input_shape_even, linear_input_shape_even)
        #     for i in range(res_dense_blocks)])

        # self.decoder_module_list_1 = nn.ModuleList([ResDenseBlock(linear_input_shape_odd * 2,
        #                                                           linear_input_shape_odd * 2)
        #                                             for i in range(res_dense_blocks)])
        #
        # self.decoder_module_list_2 = nn.ModuleList([ResDenseBlock(linear_input_shape_even * 2,
        #                                                           linear_input_shape_even * 2)
        #                                             for i in range(res_dense_blocks)])

        # self.encoder_layer_norm_1 = nn.LayerNorm(linear_input_shape_odd * 2)
        # self.encoder_layer_norm_2 = nn.LayerNorm(linear_input_shape_even * 2)

        # self.decoder_layer_norm_1 = nn.LayerNorm(linear_input_shape_odd * 2)
        # self.decoder_layer_norm_2 = nn.LayerNorm(linear_input_shape_even * 2)

        self.output_layer = nn.Linear(
            (linear_input_shape_odd + linear_input_shape_even),
            self.output_features
        )
    #
    # def encoder(self, x):
    #
    #     first_layer_1, first_layer_2 = x_1.clone(), x_2.clone()
    #
    #     for layer in self.encoder_module_list_1:
    #         x_1 = layer(x_1, first_layer_1)
    #
    #     for layer in self.encoder_module_list_2:
    #         x_2 = layer(x_2, first_layer_2)

        # x_1 = x_1.permute(2, 0, 1)
        # x_1 = self.encoder_hierarchical_lstm_layers[0](x_1)[0]
        # x_1 = torch.squeeze(x_1)
        #
        # x_2 = x_2.permute(2, 0, 1)
        # x_2 = self.encoder_hierarchical_lstm_layers[1](x_2)[0]
        # x_2 = torch.squeeze(x_2)

        # return x_1, x_2

    # def decoder(self, x_1, x_2):
    #     x_1 = torch.unsqueeze(x_1, 0)
    #     x_1 = self.decoder_hierarchical_lstm_layers[0](x_1)[0]
    #     x_1 = torch.squeeze(x_1)
    #
    #     x_2 = torch.unsqueeze(x_2, 0)
    #     x_2 = self.decoder_hierarchical_lstm_layers[1](x_2)[0]
    #     x_2 = torch.squeeze(x_2)
    #
    #     first_layer_1, first_layer_2 = x_1.clone(), x_2.clone()
    #
    #     for layer in self.decoder_module_list_1:
    #         x_1 = layer(x_1, first_layer_1)
    #
    #     for layer in self.decoder_module_list_2:
    #         x_2 = layer(x_2, first_layer_2)

        # return x_1, x_2

    def forward(self, x):
        first_layer = self.input_layer_norm(x)
        x_1, x_2 = self.encoder_hierarchical_layer(first_layer)
        x_1, x_2 = self.weighted_encoder(x_1, x_2)
        # x_1 = self.encoder_layer_norm_1(x_1)
        # x_2 = self.encoder_layer_norm_2(x_2)
        # x_1, x_2 = self.decoder(x_1, x_2)
        # x_1 = self.decoder_layer_norm_1(x_1)
        # x_2 = self.decoder_layer_norm_2(x_2)
        x = torch.concat((x_1, x_2), dim=-1)

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
            res_dense_blocks=1,
            random_seed=42
    ):
        assert in_features > 1, "in_features must be greater than 1."
        seed_everything(random_seed)
        self.out_features, self.in_features = out_features, in_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = None, None, None

        self.res_dense_blocks = res_dense_blocks

        self.model, self.loss_fn, self.optimizer = self.call(self.in_features)

    def call(self, in_features):
        model = WeightedDenseRNNBase(in_features, self.out_features, self.res_dense_blocks)
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
