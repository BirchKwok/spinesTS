import os
import numpy as np
import random

import torch


def seed_everything(seed=0):
    """Set random seed for everything.
        Include python built-in random module, numpy module, python hash seed,
        pytorch manual_seed and cuda manual_seed_all.

    Parameters
    ----------
    seed : int, random seed

    Returns
    -------
    None

    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrozenDict(dict):
    def __setitem__(self, key, value):
        raise AttributeError("FrozenDict can not be modified.")
