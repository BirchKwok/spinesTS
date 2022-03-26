import torch
from torch import nn


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

    def __init__(self, level=0.1, device=None):
        super(GaussianNoise1d, self).__init__()
        self.device = device
        self.level = level

    def forward(self, x):
        return x + torch.mul(torch.randn_like(x, device=self.device), 1 / torch.median(x) * self.level)


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

    def __init__(self, p, trainable=True, device=None):
        assert 0 <= p <= 1, "p must be greater than or equal to 0 and less than or equal to 1."
        super(TrainableDropout, self).__init__()
        self.device = device
        p = torch.Tensor([p])
        if trainable:
            self.p = nn.Parameter(p)
        else:
            self.p = p.to(self.device)

    def forward(self, x):
        assert 0 <= self.p <= 1
        # In this case, all elements are discarded
        if self.p == 1:
            return torch.zeros_like(x, device=self.device)
        # In this case, all elements are retained
        if self.p == 0:
            return x
        mask = (torch.rand(x.shape, device=self.device) > self.p).float()
        return mask * x / (1.0 - self.p)
