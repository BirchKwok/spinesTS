import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import *
from matplotlib.figure import Figure


def plot1d(*S: Sequence, x: Optional[Sequence] = None,
           title=None,
           labels: Optional[Sequence] = None,
           figsize: Sequence = (12, 8),
           legend: bool = True, linestyle='-', colors=None,
           legend_loc='best'
           ) -> Figure:
    """Plot series, accept one dim array-like sequence

    Parameters
    ----------
    S: array-like series, which to plot.
    x: None or array-like series, x-axis.
    title: figure title
    labels: None or List[str] or str-type, figure-label.
    figsize: tuple of integers, figure size
    legend: bool, whether to show the figure legend
    linestyle: line style, str-like
    legend_loc: str or pair of floats, default: rcParams["legend.loc"] (default: 'best'), it can be
        'upper left', 'upper right', 'lower left', 'lower right',
        'upper center', 'lower center', 'center left', 'center right',
        'center', 'best'

    Returns
    -------
    matplotlib.figure.Figure
    """

    if isinstance(colors, str) or colors is None:
        colors = [colors for i in range(len(S))]
    else:
        assert isinstance(colors, (list, tuple))
        assert len(colors) == len(S)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_title(title)
    seq_len = len(S)

    for i in range(seq_len):
        if labels is not None:
            if isinstance(labels, str):
                label = labels
            else:
                assert len(labels) == seq_len, "labels length must be equals to sequence length when labels is a list"
                label = labels[i]
        else:
            label = 'label_' + str(i + 1)

        if x is None:
            ax.plot(S[i], label=label, linestyle=linestyle, color=colors[i])
        else:
            ax.plot(x, S[i], label=label, linestyle=linestyle, color=colors[i])

        if legend:
            plt.legend(loc=legend_loc)

    fig.tight_layout()

    return ax


def plot2d(*S: Sequence, x: Optional[Sequence] = None,
           title=None, linestyle='-',
           labels: Optional[Sequence[str]] = None,
           eval_slices: str = '[-1]',
           figsize: Tuple[int, int] = (10, 6),
           legend: bool = True,
           subplots_shape: Tuple[int, int] = (-1, 5), colors=None,
           legend_loc='best'
           ):
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
    title: figure title, str-like
    linestyle: line style, str-like
    colors: line colors, None or str-like or list, or tuple
    legend_loc: str or pair of floats, default: rcParams["legend.loc"] (default: 'best'), it can be
        'upper left', 'upper right', 'lower left', 'lower right',
        'upper center', 'lower center', 'center left', 'center right',
        'center', 'best'

    Returns
    -------
    matplotlib.figure.Figure
    """
    assert np.sum([np.ndim(i) == 2 for i in S]) == len(S)
    assert isinstance(eval_slices, str)

    if isinstance(colors, str) or colors is None:
        colors = [colors for i in range(len(S))]
    else:
        assert isinstance(colors, (list, tuple))
        assert len(colors) == len(S)

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

    for j in range(len(_to_plot_array)):  # figures
        for i in range(1, num + 1):
            ax = plt.subplot(rows, cols, i)
            if title is not None:
                ax.set_title(title+f' {i-1}')

            if labels is None:
                label = f'label_' + str(j + 1)
            else:
                label = labels[j]
            if x is not None:
                ax.plot(x, _to_plot_array[j][i - 1], label=label, linestyle=linestyle, color=colors[j])
            else:
                ax.plot(_to_plot_array[j][i - 1], label=label, linestyle=linestyle, color=colors[j])
            ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
            if legend:
                ax.legend(loc=legend_loc)

    fig.tight_layout()
    return fig
