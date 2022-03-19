import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot1d(*s, figsize=(12, 8)):
    """Plot series"""
    plt.figure(figsize=figsize)
    for i in range(len(s)):
        plt.plot(s[i], label='label_' + str(i + 1))
        plt.legend()

    plt.show()


def plot2d(*S, x=None, labels=None, eval_slices='[:30]',
           figsize=(10, 6), legend=True, subplots_shape=(-1, 5)):
    """
    plot multi steps.
    """
    assert np.sum([i.ndim == 2 for i in S]) == len(S)
    assert isinstance(eval_slices, str)

    fig = plt.figure(facecolor='w', figsize=figsize)

    _to_plot_array = []
    for _s in range(len(S)):
        _v = eval('S[_s]' + eval_slices)
        if not isinstance(_v[0], (list, tuple, np.ndarray, pd.Series)):
            _v = [_v]
        _to_plot_array.append(_v)

    num = max([len(_) for _ in _to_plot_array])

    cols = subplots_shape[-1] if num >= 5 else 1
    rows = num if num < 5 else (num // subplots_shape[-1] + 1 if num % subplots_shape[-1] > 0
                                else num // subplots_shape[-1])

    for j in range(len(_to_plot_array)):
        for i in range(1, num + 1):
            ax = plt.subplot(rows, cols, i)
            if labels is None:
                label = f'label_' + str(j + 1)
            else:
                label = labels[j]
            if x is not None:
                ax.plot(x, _to_plot_array[j][i-1], label=label, ls='-')
            else:
                ax.plot(_to_plot_array[j][i-1], label=label, ls='-')
            if legend:
                ax.legend()

    fig.tight_layout()
    return fig
