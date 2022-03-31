import torch
from torch import nn


class ResBlock(nn.Module):
    """Residual connection block.

    Add two tensors of the same shape passed in and return the sum of the two.

    Parameters
    ----------
    trainable : bool, If true, the element with index 1 is multiplied by a trainable scalar,
        followed by a residual join.This can be thought of as taking into account
        the join ratio of elements with index 1.

    Returns
    -------
    torch.Tensor
    """
    def __init__(self, trainable=True):
        super(ResBlock, self).__init__()
        self.coefficient_1 = nn.Parameter(torch.Tensor([1.]))
        self.coefficient_2 = nn.Parameter(torch.Tensor([1.]))
        self.trainable = trainable

    def forward(self, inputs):
        assert len(inputs) == 2, "ResBlock only take a sequence with two elements."
        if self.trainable:
            return inputs[0] * self.coefficient_1 + inputs[1] * self.coefficient_2
        else:
            return inputs[0] + inputs[1]
