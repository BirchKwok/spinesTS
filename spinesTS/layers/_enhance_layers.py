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
