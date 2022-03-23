import torch
from torch import nn


class Hierarchical1d(nn.Module):
    """
    Splits the incoming sequence into two sequences by parity index.

    Returns
    -------
    torch.Tensor
    """
    def __init__(self):
        super(Hierarchical1d, self).__init__()

    @staticmethod
    def even(x):
        return x[:, ::2]

    @staticmethod
    def odd(x):
        return x[:, 1::2]

    def forward(self, x):
        """Returns the odd and even part"""
        return self.even(x), self.odd(x)


class TrainableMovingAverage1d(nn.Module):
    """Take the moving average of the incoming sequence,
    accept a 2-dimensional tensor x,
    and return a 2-dimensional tensor of the shape:
    (x.shape[0], x.shape[1] // kernel_size + x.shape[1] % kernel_size)

    Parameters
    ----------
    kernel_size: int, moving average window size
    weighted: bool, if true, it is a weighted moving average and the weight is a trainable parameter;
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
        res = torch.empty((rows, cols // self.kernel_size + cols % self.kernel_size))
        for i in range(rows):
            for j in range(cols // self.kernel_size + cols % self.kernel_size):
                if j * self.kernel_size < cols:
                    _ = x[i, j * self.kernel_size: (j+1) * self.kernel_size]
                    if len(_) < self.kernel_size:
                        _ = torch.concat((_, torch.zeros(self.kernel_size - len(_))))
                    if self.weighted:
                        res[i, j] = torch.sum(torch.mul(_, self.weighted))
                    else:
                        res[i, j] = torch.mean(_)
        return res


class SeasonalLayer(nn.Module):
    """

    Parameters
    ----------
    s: int,
    trainable: bool,


    Returns
    -------
    torch.Tensor

    """
    def __init__(self, s, trainable=True):
        super(SeasonalLayer, self).__init__()
        self.s = s
        self.trainable = trainable

    def forward(self, x):
        pass



