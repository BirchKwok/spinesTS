import numpy as np


def _base_op(y_true, y_pred, func):
    """
    y_true and y_pred must be (n_samples, ) or (n_samples, n_features) shape sequence
    """
    assert y_true.ndim <= 2, f"Found array with dim {y_true.ndim}, expected <= 2."

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape == y_pred.shape:
        return np.array(list(map(func, [(y_true[i], y_pred[i]) for i in range(y_true.shape[0])])))
    else:
        raise ValueError("sequences shape must be the same.")


def mean_absolute_error(y_true, y_pred):
    """
    ```math
    \frac{\sum_{i=1}^{n}{|y\_true_i - y\_pred_i}|}{n}
    ```
    mean absolute error.
    """

    _ = lambda ys: np.abs(ys[0] - ys[1])
    return np.sum(_base_op(y_true, y_pred, _)) / np.asarray(y_true).size


def mean_absolute_percentage_error(y_true, y_pred, ignore_zeros=True):
    """
    mean absolute percentage error
    """
    def _(ys):
        yt, yp = ys
        
        if not ignore_zeros:
            epsilon = np.finfo(np.float64).eps
            _ = np.abs(yt - yp) / np.maximum(np.abs(yt), epsilon)
        else:
            _ = np.abs(yt - yp) / np.abs(yt)

        return np.where(np.isnan(_), np.inf, _)

    res = _base_op(y_true, y_pred, _)

    res = [j for i in res for j in i if not np.isinf(j)]

    return np.mean(res)


