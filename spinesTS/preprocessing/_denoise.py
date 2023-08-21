import pandas as pd


def moving_average(x, window_size=3):
    assert window_size > 1
    if x.ndim == 1:
        x = pd.Series(x)
        return x.rolling(window_size).mean().values[window_size-1:]
    elif x.ndim == 2:
        x = pd.DataFrame(x.T)
        return x.rolling(window_size).mean().values.T[:, window_size-1:]
    else:
        raise ValueError("x must be one dim or two dim sequence.")
    