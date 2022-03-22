import torch
from torch import nn


class Hierarchical1d(nn.Module):
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


class MovingAverage1d(nn.Module):
    def __init__(self, kernel_size, weighted=True):
        super(MovingAverage1d, self).__init__()
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
    def __init__(self):
        super(SeasonalLayer, self).__init__()

    def forward(self, x):
        pass



