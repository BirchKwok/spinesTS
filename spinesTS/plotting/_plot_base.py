import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import *
from matplotlib.figure import Figure


def plot1d(*S: Sequence, x: Optional[Sequence] = None,
           labels: Optional[Sequence] = None,
           figsize: Sequence = (12, 8),
           legend: bool = True) -> Figure:
    """Plot series, accept one dim array-like sequence

    Parameters
    ----------
    S: array-like series, which to plot.
    x: None or array-like series, x-axis.
    labels: None or List[str], figure-label.
    figsize: tuple of integers, figure size
    legend: bool, whether to show the figure legend

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    for i in range(len(S)):
        if labels is not None:
            label = labels[i]
        else:
            label = 'label_' + str(i + 1)

        if x is None:
            plt.plot(S[i], label=label)
        else:
            plt.plot(x, S[i], label=label)

        if legend:
            plt.legend()

    return fig


def plot2d(*S: Sequence, x: Optional[Sequence] = None,
           labels: Optional[Sequence[str]] = None,
           eval_slices: str = '[:30]',
           figsize: Tuple[int, int] = (10, 6),
           legend: bool = True,
           subplots_shape: Tuple[int, int] = (-1, 5)):
    """Plot series, accept two dim array-like sequence

    Parameters
    ----------
    S: array-like series, which to plot.
    x: None or array-like series, x-axis.
    labels: None or List[str], figure-label.
    eval_slices: str,
    figsize: tuple of integers, figure size
    legend: bool, whether to show the figure legend
    subplots_shape: tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    assert np.sum([np.ndim(i) == 2 for i in S]) == len(S)
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
                ax.plot(x, _to_plot_array[j][i - 1], label=label, ls='-')
            else:
                ax.plot(_to_plot_array[j][i - 1], label=label, ls='-')
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            if legend:
                ax.legend()

    fig.tight_layout()
    return fig
