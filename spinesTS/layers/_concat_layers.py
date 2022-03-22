import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.coefficient = nn.Parameter(torch.Tensor([1.]))

    def forward(self, inputs):
        return inputs[0] + inputs[1] * self.coefficient
