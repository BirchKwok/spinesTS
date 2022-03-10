import numpy as np
import torch
from torch import nn

from spinesTS.utils import seed_everything
from spinesTS.base import TorchModelMixin


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.coefficient = nn.Parameter(torch.Tensor([1.]))

    def forward(self, inputs):
        return inputs[0] + inputs[1] * self.coefficient


class HierarchicalLayer(nn.Module):
    def __init__(self):
        super(HierarchicalLayer, self).__init__()

    @staticmethod
    def even(x):
        return x[:, ::2]

    @staticmethod
    def odd(x):
        return x[:, 1::2]

    def forward(self, x):
        """Returns the odd and even part"""
        return self.even(x), self.odd(x)


class ResDenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResDenseBlock, self).__init__()
        self.fc_block_1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features),
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features),
        )
        self.res_layer_1 = ResBlock()
        self.res_layer_2 = ResBlock()
        self.last_res_layer = ResBlock()

    def forward(self, x, init_layer):
        # block 1
        x = self.fc_block_1(x)
        x = torch.squeeze(x)
        x = self.res_layer_1([init_layer, x])

        up_level_layer = x.clone()

        # block 2
        x = self.fc_block_2(x)
        x = self.last_res_layer([init_layer, self.res_layer_2([up_level_layer, x])])

        return x


class MLPTorch(nn.Module):
    def __init__(self, in_features, output_features, res_dense_blocks=1):
        super(MLPTorch, self).__init__()
        self.in_features, self.output_features = in_features, output_features
        self.input_layer_norm = nn.LayerNorm(self.in_features)
        self.encoder_hierarchical_layer = HierarchicalLayer()

        linear_input_shape = int(np.ceil(self.in_features / 2))
        self.encoder_hierarchical_lstm_layers = nn.ModuleList([
            nn.LSTM(linear_input_shape, linear_input_shape, 1, bidirectional=True) for i in range(2)
        ])

        self.decoder_hierarchical_lstm_layers = nn.ModuleList([
            nn.LSTM(self.in_features, linear_input_shape, 1, bidirectional=True) for i in range(2)
        ])

        self.encoder_module_list_1 = nn.ModuleList([ResDenseBlock(linear_input_shape, linear_input_shape)
                                                    for i in range(res_dense_blocks)])
        self.encoder_module_list_2 = nn.ModuleList([ResDenseBlock(linear_input_shape, linear_input_shape)
                                                    for i in range(res_dense_blocks)])

        self.decoder_module_list_1 = nn.ModuleList([ResDenseBlock(self.in_features, out_features=self.in_features)
                                                    for i in range(res_dense_blocks)])
        self.decoder_module_list_2 = nn.ModuleList([ResDenseBlock(self.in_features, out_features=self.in_features)
                                                    for i in range(res_dense_blocks)])

        self.encoder_layer_norm = nn.LayerNorm(self.in_features*2)
        self.decoder_layer_norm = nn.LayerNorm(self.in_features*2)

        self.output_layer = nn.Linear(self.in_features * 2, self.output_features)

    def encoder(self, x):
        x_1, x_2 = self.encoder_hierarchical_layer(x)
        first_layer_1, first_layer_2 = x_1.clone(), x_2.clone()

        for layer in self.encoder_module_list_1:
            x_1 = layer(x_1, first_layer_1)

        for layer in self.encoder_module_list_2:
            x_2 = layer(x_2, first_layer_2)

        x_1 = torch.unsqueeze(x_1, 0)
        x_1 = self.encoder_hierarchical_lstm_layers[0](x_1)[0]
        x_1 = torch.squeeze(x_1)

        x_2 = torch.unsqueeze(x_2, 0)
        x_2 = self.encoder_hierarchical_lstm_layers[1](x_2)[0]
        x_2 = torch.squeeze(x_2)

        return torch.concat((x_1, x_2), dim=1)  # in_features * 2

    def decoder(self, x):
        x_1, x_2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]
        first_layer_1, first_layer_2 = x_1.clone(), x_2.clone()

        for layer in self.decoder_module_list_1:
            x_1 = layer(x_1, first_layer_1)

        for layer in self.decoder_module_list_2:
            x_2 = layer(x_2, first_layer_2)

        x_1 = torch.unsqueeze(x_1, 0)
        x_1 = self.decoder_hierarchical_lstm_layers[0](x_1)[0]
        x_1 = torch.squeeze(x_1)

        x_2 = torch.unsqueeze(x_2, 0)
        x_2 = self.decoder_hierarchical_lstm_layers[1](x_2)[0]
        x_2 = torch.squeeze(x_2)

        return torch.concat((x_1, x_2), dim=1)

    def forward(self, x):
        first_layer = self.input_layer_norm(x)
        x = self.encoder(first_layer)
        x = self.encoder_layer_norm(x)
        x = self.decoder(x)
        x = self.decoder_layer_norm(x)

        return self.output_layer(x)


class MLPTorchModel(TorchModelMixin):
    """
    spinesTS MLP pytorch-model

    """

    def __init__(self, in_features, output_nums, learning_rate=0.001, res_dense_blocks=1,
                 random_seed=0):
        seed_everything(random_seed)
        self.output_nums, self.in_features = output_nums, in_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = None, None, None

        self.res_dense_blocks = res_dense_blocks

        self.model, self.loss_fn, self.optimizer = self.call(self.in_features)

    def call(self, in_features):
        model = MLPTorch(in_features, self.output_nums, self.res_dense_blocks)
        loss_fn = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(self, X_train, y_train, epochs=1000, batch_size='auto', eval_set=None,
            monitor='val_loss', min_delta=0, patience=10,
            restore_best_weights=True, verbose=True):
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        return self._fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                         monitor=monitor,
                         min_delta=min_delta, patience=patience, restore_best_weights=restore_best_weights,
                         verbose=verbose)

    def predict(self, x):
        assert self.model is not None, "model not fitted yet."
        return self._predict(x)


