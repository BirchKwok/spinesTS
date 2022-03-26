import torch


def one_dim_tensor_del_elements(tensor, start_index, end_index=None):
    """Delete elements from a one dim torch tensor.
    
    Parameters
    ----------
    tensor: torch.Tensor, with one dim
    start_index: int, index of the value to be dropped
    end_index: int, index of the value to be dropped, if not None,
    the tensor will be dropped from the start_index to the end_index
    
    Returns
    -------
    torch.Tensor
    """
    assert isinstance(start_index, int)
    assert isinstance(end_index, int) or end_index is None
    if end_index is None:
        if start_index == 0:
            return tensor[1:]
        elif start_index == -1 or start_index == len(tensor):
            return tensor[:-1]
        else:
            return torch.concat((tensor[:start_index], tensor[start_index+1:]))
    else:
        assert end_index >= start_index
        return torch.concat((tensor[:start_index], tensor[end_index+1:]))
