def get_weight_norm(device):
    """
    Weight normalization implementation
    """
    import torch
    from spinesTS.base._torch_mixin import detect_available_device

    if detect_available_device(device)[0] == 'mps':
        from .weight_norm import weight_norm
        return weight_norm
    else:
        return torch.nn.utils.weight_norm


def get_remove_weight_norm(device):
    import torch
    from spinesTS.base._torch_mixin import detect_available_device

    if detect_available_device(device)[0] == 'mps':
        from .weight_norm import remove_weight_norm
        return remove_weight_norm
    else:
        return torch.nn.utils.remove_weight_norm


