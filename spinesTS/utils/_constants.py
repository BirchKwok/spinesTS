import os
import random
from typing import Optional
import warnings

import numpy as np
import torch

from ..frame import DataTS

MIN_SEED_VALUE = 0
MAX_SEED_VALUE = 2**32 - 1


def seed_everything(seed: Optional[int] = None) -> int:
    """
    Set a random seed for reproducibility across various libraries.

    Parameters
    ----------
    seed : int, optional
        The random seed. If None, a random seed is generated. Default is None.

    Returns
    -------
    int
        The random seed used.
    """
    if seed is None:
        seed = np.random.randint(MIN_SEED_VALUE, MAX_SEED_VALUE)
        warnings.warn(f"No seed specified, using randomly generated seed: {seed}")
    else:
        try:
            seed = int(seed)
            if not (MIN_SEED_VALUE <= seed <= MAX_SEED_VALUE):
                raise ValueError
        except ValueError:
            seed = np.random.randint(MIN_SEED_VALUE, MAX_SEED_VALUE)
            warnings.warn(f"Invalid seed '{seed}', using randomly generated seed.")

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    return seed


def check_if_datats(x):
    """check input frame type"""
    if not isinstance(x, DataTS):
        raise TypeError("Only accept dataframe input of type spinesTS.frame.DataTS, "
                        "maybe you can try spinesTS.DataTS(your_pd_frame) to fix that")
    return