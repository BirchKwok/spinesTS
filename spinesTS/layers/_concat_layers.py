import torch
from torch import nn


class RecurseResBlock(nn.Module):
    """Recursion Residual connection block.

    Add two tensors of the same shape passed in and return the sum of the two.

    Parameters
    ----------
    in_nums : int, inputs_numbers + 1
    trainable : bool, If true, the element with index 1 is multiplied by a trainable scalar,
        followed by a residual join.This can be thought of as taking into account
        the join ratio of elements with index 1.

    Returns
    -------
    torch.Tensor
    """
    def __init__(self, in_nums, trainable=True):
        super(RecurseResBlock, self).__init__()
        self.in_nums = in_nums

        self.trainable = trainable
        if self.trainable:
            self.coefficient_1 = nn.Parameter(torch.Tensor([1.]))
            self.coefficient_2 = nn.Parameter(torch.Tensor([1.]))

        if self.in_nums != 2:
            self.sub_block = RecurseResBlock(self.in_nums - 1, trainable=self.trainable)

    def forward(self, inputs):
        assert len(inputs) == self.in_nums
        if self.trainable:
            if self.in_nums != 2:
                return inputs[0] * self.coefficient_1 + self.sub_block(inputs[1:]) * self.coefficient_2
            else:
                return inputs[0] * self.coefficient_1 + inputs[1] * self.coefficient_2
        else:
            if self.in_nums != 2:
                return inputs[0] + self.sub_block(inputs[1:])
            else:
                return inputs[0] + inputs[1]
