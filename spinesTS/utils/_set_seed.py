import os
import numpy as np
import random

import torch


def seed_everything(seed=None, tf_seed=False):

    random.seed(seed)
    np.random.seed(seed)
    if tf_seed:
        import tensorflow as tf
        tf.random.set_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
