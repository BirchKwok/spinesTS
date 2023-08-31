import random
from typing import Optional
import warnings

import numpy as np
import torch

from ..frame import DataTS


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    """set random seed for everything.

    Parameters
    ----------
    seed: int, random seed, default None

    Returns
    -------
    int, random seed after setting.
    """
    if seed is None:
        seed = np.random.randint(min_seed_value, max_seed_value)
        warnings.warn(f"No seed found, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        warnings.warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = np.random.randint(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def check_if_datats(x):
    """check input frame type"""
    if not isinstance(x, DataTS):
        raise TypeError("Only accept dataframe input of type spinesTS.frame.DataTS, "
                        "maybe you can try spinesTS.DataTS(your_pd_frame) to fix that")
    return