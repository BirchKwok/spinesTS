import random
from typing import Optional
import warnings

import numpy as np
import torch

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed: Optional[int] = None) -> int:
    """
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


class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise AttributeError("FrozenDict can not be modified.")
