import torch
from torch import nn


class Hierarchical1d(nn.Module):
    """
    Splits the incoming sequence into two sequences by parity index.

    Returns
    -------
    torch.Tensor, even, odd
    """

    def __init__(self):
        super(Hierarchical1d, self).__init__()

    @staticmethod
    def odd(x):
        return x[:, ::2]

    @staticmethod
    def even(x):
        return x[:, 1::2]

    def forward(self, x):
        """Returns the odd and even part"""
        return self.odd(x), self.even(x)


class HierarchicalEmbedding(nn.Module):
    def __init__(self, in_features, kernel=5, dropout=0.5, groups=1, hidden_size=1):
        super(HierarchicalEmbedding, self).__init__()
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1  # by default: stride==1
            pad_r = self.dilation * self.kernel_size // 2 + 1  # by default: stride==1
        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1  # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_features * prev_size, int(in_features * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_features * size_hidden), in_features,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_features * prev_size, int(in_features * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_features * size_hidden), in_features,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_features * prev_size, int(in_features * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_features * size_hidden), in_features,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_features * prev_size, int(in_features * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups=self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_features * size_hidden), in_features,
                      kernel_size=3, stride=1, groups=self.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x_even, x_odd):

        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        d = x_odd - self.P(x_even)
        c = x_even + self.U(d)

        return c, d


class TrainableMovingAverage1d(nn.Module):
    """Take the moving average of the input sequence.
        accept a 2-dimensional tensor x, and return a 2-dimensional tensor of the shape:
            (x.shape[0], x.shape[1] - kernel_size)

    Parameters
    ----------
    kernel_size : int, moving average window size
    weighted : bool, if true, it is a weighted moving average and the weight is a trainable parameter;
        otherwise, it is a simple moving average
    Returns
    -------
    torch.Tensor
    """

    def __init__(self, kernel_size, weighted=True):
        super(TrainableMovingAverage1d, self).__init__()
        self.kernel_size = kernel_size
        if weighted:
            self.weighted = nn.Parameter(torch.randn(self.kernel_size))
        else:
            self.weighted = None

    def forward(self, x):
        assert x.ndim == 2, "MovingAverageLayer accept a two dims input."
        rows, cols = x.shape
        col = cols - self.kernel_size

        res = torch.empty((rows, col), device=x.device)
        for i in range(rows):
            for j in range(cols - self.kernel_size):
                _ = j + self.kernel_size
                if _ < cols:
                    _2 = x[i, j: _]
                    if self.weighted is not None:
                        res[i, j: _] = torch.sum(torch.mul(_2, self.weighted))
                    else:
                        res[i, j: _] = torch.mean(_2)
        return res
