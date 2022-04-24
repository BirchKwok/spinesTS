import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functools import partial


def split_series(x_seq, y_seq, window_size, pred_steps, train_size=None, shuffle=False, skip_steps=1):
    """
    return :
    when train_size is not None, return X_train, X_test, y_train, y_test
    otherwise return X, y
    """
    assert isinstance(x_seq, (pd.Series, np.ndarray, list)) and isinstance(y_seq, (pd.Series, np.ndarray, list))
    assert train_size is None or (0< train_size < 1 and isinstance(train_size, float))
    assert isinstance(shuffle, bool)
    assert isinstance(pred_steps, int) and pred_steps > 0

    if isinstance(x_seq, pd.Series):
        x_seq = x_seq.values
    elif isinstance(x_seq, list):
        x_seq = np.array(x_seq)
    if isinstance(y_seq, pd.Series):
        y_seq = y_seq.values
    elif isinstance(y_seq, list):
        y_seq = np.array(y_seq)
    X, y = [], []

    for i in range(len(x_seq)):
        end_index = i * skip_steps + window_size
        out_end_index = end_index + pred_steps

        if out_end_index > len(x_seq):
            break

        seq_x, seq_y = list(x_seq[i*skip_steps:end_index]), list(y_seq[end_index:out_end_index])

        X.append(seq_x)
        y.append(seq_y)

    if pred_steps == 1 and train_size is None:
        return np.array(X), np.squeeze(np.array(y))
    elif pred_steps == 1 and train_size is not None:
        return train_test_split(np.array(X), np.squeeze(np.array(y)), train_size=train_size, shuffle=shuffle)
    elif pred_steps > 1 and train_size is None:
        return np.array(X), np.array(y)
    else:
        return train_test_split(np.array(X), np.array(y), train_size=train_size, shuffle=shuffle)


train_test_split_ts = partial(train_test_split, shuffle=False)  # train_test_split for time series
