import torch
from torch import nn
from spinesTS.layers import RecurseResBlock


class GaussianNoise1d(nn.Module):
    """Adds Gaussian noise to the input tensor.

    Parameters
    ----------
    level: float, the scaling multiple of gaussian noise
    device: str or None, device name

    Returns
    -------
    torch.Tensor
    """

    def __init__(self, level=0.1):
        super(GaussianNoise1d, self).__init__()
        self.level = level

    def forward(self, x):
        return x + torch.mul(torch.randn_like(x, device=x.device), 1 / torch.median(x) * self.level)


class ResDenseBlock(nn.Module):
    def __init__(self, in_features, kernel_size=5, dilation=1):
        super(ResDenseBlock, self).__init__()
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.dilation = dilation
        # if self.kernel_size % 2 == 0:
        #     pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
        #     pad_r = self.dilation * self.kernel_size // 2 + 1  # by default: stride==1
        # else:
        #     pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
        #     pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1

        self.fc_blocks = nn.ModuleList([
            nn.Sequential(
                # nn.Linear(in_features, in_features),
                # nn.LayerNorm(in_features),
                # nn.LeakyReLU(),
                # nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(in_features, in_features,
                          kernel_size=self.kernel_size, dilation=self.dilation, stride=1,
                          padding='same'),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(0.3),
                nn.Conv1d(in_features, in_features,
                          kernel_size=3, stride=1, padding='same'),
                nn.Tanh()
            ) for i in range(2)
        ])
        self.res_layer_1 = RecurseResBlock(2, trainable=True)
        self.last_res_layer = RecurseResBlock(3, trainable=True)

    def forward(self, x, init_layer):
        # block 1
        x = self.fc_blocks[0](x.view(-1, self.in_features, 1))
        x = self.res_layer_1([init_layer, x.view(-1, self.in_features)])

        _ = x.clone()

        # block 2
        x = self.fc_blocks[1](x.view(-1, self.in_features, 1))
        x = self.last_res_layer([init_layer, _, x.view(-1, self.in_features)])

        return x


class Time2Vec(nn.Module):
    """Time2Vec module.

    Parameters
    ----------
    in_features: int, input-tensor last dimension shape

    Returns
    -------
    None
    """

    def __init__(self, in_features):
        super(Time2Vec, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, in_features))
        self.P = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(1))
        self.p = nn.Parameter(torch.randn(1))
        self.sine_w = nn.Parameter(torch.randn(1))
        self.cosine_w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        original = self.w * x + self.p
        x = torch.matmul(x, self.W)
        for i in range(x.shape[0]):
            x[i, :] = self.sine_w * torch.sin(torch.squeeze(x[i]) + self.P) + \
                      self.cosine_w * torch.cos(torch.squeeze(x[i]) + self.P)

        return torch.concat((original, x), dim=-1)  # last dimension shape (, 2 * in_features)