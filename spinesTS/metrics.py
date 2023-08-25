import numpy as np
from sklearn.metrics import *
from torch import nn
import torch


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


def rmse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs) ** 0.5


def mae(*args, **kwargs):
    return mean_absolute_error(*args, **kwargs)


def mse(*args, **kwargs):
    return mean_squared_error(*args, **kwargs)


class WMAPELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(WMAPELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.abs(inputs - targets).sum() / torch.abs(inputs).sum()


class RMSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return torch.sqrt(self.mse(inputs, targets))