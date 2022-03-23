import torch
from torch import nn


class GaussianNoise1d(nn.Module):
    """Adds Gaussian noise to the input tensor.

    Returns
    -------
    torch.Tensor
    """
    def __init__(self):
        super(GaussianNoise1d, self).__init__()

    def forward(self, x):
        return x + torch.randn_like(x)


class TrainableDropout(nn.Module):
    """Dropout layer with a trainable dropout ratio.

    Parameters
    ----------
    p: float, dropout ratio, must be greater than or equal to 0 and less than or equal to 1
    trainable: bool, set the parameter p to trainable status

    Returns
    -------
    torch.Tensor
    """
    def __init__(self, p, trainable=True):
        assert 0 <= p <= 1, "p must be greater than or equal to 0 and less than or equal to 1."
        super(TrainableDropout, self).__init__()
        if trainable:
            self.p = nn.Parameter(p)
        else:
            self.p = p

    def forward(self, x):
        assert 0 <= self.p <= 1
        # 在本情况中，所有元素都被丢弃
        if self.p == 1:
            return torch.zeros_like(x)
        # 在本情况中，所有元素都被保留
        if self.p == 0:
            return x
        mask = (torch.rand(x.shape) > self.p).float()
        return mask * x / (1.0 - self.p)
