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
    """Take the moving average of the input sequence.
        accept a 2-dimensional tensor x, and return a 2-dimensional tensor of the shape:
            (x.shape[0], x.shape[1] - kernel_size)

    Parameters
    ----------
    kernel_size : int, moving average window size
    weighted : bool, if true, it is a weighted moving average and the weight is a trainable parameter;
        otherwise, it is a simple moving average
    device : torch Tensor device.
    Returns
    -------
    torch.Tensor
    """

    def __init__(self, kernel_size, weighted=True, device=None):
        super(TrainableMovingAverage1d, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        if weighted:
            self.weighted = nn.Parameter(torch.randn(self.kernel_size))
        else:
            self.weighted = None

    def forward(self, x):
        assert x.ndim == 2, "MovingAverageLayer accept a two dims input."
        rows, cols = x.shape
        col = cols - self.kernel_size

        res = torch.empty((rows, col), device=self.device)
        for i in range(rows):
            for j in range(cols-self.kernel_size):
                _ = j + self.kernel_size
                if _ < cols:
                    _2 = x[i, j: _]
                    if self.weighted is not None:
                        res[i, j: _] = torch.sum(torch.mul(_2, self.weighted))
                    else:
                        res[i, j: _] = torch.mean(_2)
        return res


class SeasonalLayer1D(nn.Module):
    def __init__(self):
        super(SeasonalLayer1D, self).__init__()

    def forward(self, x):
        pass
