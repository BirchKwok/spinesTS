import numpy as np


def check_x_y(x, y):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert np.ndim(x) <= 3 and np.ndim(y) <= 2
    assert len(x) == len(y)
