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


class MoveAvg(nn.Module):
    """Take the moving average of the input sequence.
            accept a 2-dimensional tensor x, and return a 2-dimensional tensor of the shape:
                (x.shape[0], x.shape[1] - kernel_size)

    Parameters
    ----------
    kernel_size : int, moving average window size
    stride : int, the jump-step-length in calculating the moving average
    padding : str, only support to 'same' and 'valid', valid means not to pad, same means padding with
    the nearest valid value to the null value
    Returns
    -------
    torch.Tensor
    """
    def __init__(self, kernel_size, stride=1, padding='neighbor'):
        assert padding in ('neighbor', 'valid')
        super(MoveAvg, self).__init__()
        self.kernel_size, self.stride = kernel_size, stride
        self.padding = padding
        # (, , x.shape[-1] - kernel_size + 1 - stride)
        self.moving_layer = nn.AvgPool1d(kernel_size, stride=stride, padding=0)

    def forward(self, x):
        moving_mean = self.moving_layer(x)
        _ = torch.cat((moving_mean[:, :, -(self.kernel_size - 1 + self.stride - 1):], moving_mean), dim=-1)
        res = x - _
        if self.padding == 'neighbor':
            return res, _
        return res, moving_mean


class DimensionConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, **kwargs):
        super(DimensionConv1d, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            **kwargs
        )

    def forward(self, x):
        assert x.ndim == 2
        return self.conv1d(x.view(x.shape[0], x.shape[1], 1)).squeeze()
