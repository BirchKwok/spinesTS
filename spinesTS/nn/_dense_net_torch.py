import torch
from torch import nn

from spinesTS.utils import seed_everything
from spinesTS.base import TorchModelMixin


class UnSqueeze(nn.Module):
    def __init__(self, dim=1):
        super(UnSqueeze, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.unsqueeze(inputs, dim=self.dim)


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm1d(input_channels), nn.ReLU(),
        nn.Conv1d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm1d(input_channels), nn.ReLU(),
        nn.Conv1d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool1d(kernel_size=2, stride=2))


def dense_net(out_features, kernel_size=3):
    b1 = nn.Sequential(
        UnSqueeze(1),
        nn.Conv1d(1, 64, kernel_size=kernel_size, stride=2, padding=3),
        nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

    # num_channels为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(
        b1, *blks,
        nn.BatchNorm1d(num_channels), nn.ReLU(),
        nn.AdaptiveMaxPool1d(1),  # global average pooling
        nn.Flatten(),
        nn.Linear(num_channels, out_features))

    return net


class DenseNet1DTorchModel(TorchModelMixin):
    """
    spinesTS DenseNet1D pytorch-model
    """
    def __init__(self, output_nums, kernel_size=3, learning_rate=0.001, random_seed=0):
        seed_everything(random_seed)
        self.output_nums = output_nums
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.model, self.loss_fn, self.optimizer = None, None, None

        self.model, self.loss_fn, self.optimizer = self.call()

    def call(self):
        model = dense_net(self.output_nums, kernel_size=self.kernel_size)
        loss_fn = nn.HuberLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        return model, loss_fn, optimizer

    def fit(self, X_train, y_train, epochs=1000, batch_size='auto', eval_set=None,
            monitor='val_loss', min_delta=0, patience=10,
            restore_best_weights=True, verbose=True):
        X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train)

        return self._fit(X_train, y_train, epochs, batch_size, eval_set, loss_type='down', metrics_name='mae',
                         monitor=monitor, min_delta=min_delta, patience=patience,
                         restore_best_weights=restore_best_weights, verbose=verbose)

    def predict(self, x):
        assert self.model is not None, "model not fitted yet."
        return self._predict(x)