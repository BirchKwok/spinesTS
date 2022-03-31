import numpy as np
import torch
from torch import nn

from spinesTS.utils import seed_everything
from spinesTS.base import TorchModelMixin
from spinesTS.layers import ResBlock, Hierarchical1d


class ResDenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResDenseBlock, self).__init__()
        self.fc_block_1 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.fc_block_2 = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.res_layer_1 = ResBlock()
        self.res_layer_2 = ResBlock()
        self.last_res_layer = ResBlock()

    def forward(self, x, init_layer):
        # block 1
        x = self.fc_block_1(x)
        x = self.res_layer_1([init_layer, x])

        up_level_layer = x.clone()

        # block 2
        x = self.fc_block_2(x)
        x = self.last_res_layer([init_layer, self.res_layer_2([up_level_layer, x])])

        return x


class RWDNet(nn.Module):
    def __init__(self, in_features, output_features, res_dense_blocks=1):
        super(RWDNet, self).__init__()
        self.in_features, self.output_features = in_features, output_features
        self.input_layer_norm = nn.LayerNorm(self.in_features)
        self.encoder_hierarchical_layer = Hierarchical1d()

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

        self.encoder_layer_norm_1 = nn.LayerNorm(self.in_features)
        self.encoder_layer_norm_2 = nn.LayerNorm(self.in_features)

        self.decoder_layer_norm_1 = nn.LayerNorm(self.in_features)
        self.decoder_layer_norm_2 = nn.LayerNorm(self.in_features)

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

        return x_1, x_2

    def decoder(self, x_1, x_2):
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

        return x_1, x_2

    def forward(self, x):
        first_layer = self.input_layer_norm(x)
        x_1, x_2 = self.encoder(first_layer)
        x_1 = self.encoder_layer_norm_1(x_1)
        x_2 = self.encoder_layer_norm_2(x_2)
        x_1, x_2 = self.decoder(x_1, x_2)
        x_1 = self.decoder_layer_norm_1(x_1)
        x_2 = self.decoder_layer_norm_2(x_2)
        x = torch.concat((x_1, x_2), dim=-1)

        return self.output_layer(x)


class RecurrentWeightedDenseNet(TorchModelMixin):
    """
    spinesTS MLP pytorch-model

    """

    def __init__(
            self,
            in_features,
            output_nums,
            learning_rate=0.0001,
            res_dense_blocks=1,
            random_seed=0
    ):
        assert in_features > 1, "in_features must be greater than 1."
        seed_everything(random_seed)
        self.output_nums, self.in_features = output_nums, in_features
        self.learning_rate = learning_rate
        self.model, self.loss_fn, self.optimizer = None, None, None

        self.res_dense_blocks = res_dense_blocks

        self.model, self.loss_fn, self.optimizer = self.call(self.in_features)

    def call(self, in_features):
        model = RWDNet(in_features, self.output_nums, self.res_dense_blocks)
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
