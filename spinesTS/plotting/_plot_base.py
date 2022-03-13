import matplotlib.pyplot as plt
import numpy as np


def plot1d(*s, figsize=(12, 8)):
    """Plot series"""
    plt.figure(figsize=figsize)
    for i in range(len(s)):
        plt.plot(s[i], label='label_'+str(i+1))
        plt.legend()

    plt.show()


def plot2d(*s, labels=None, fig_num_or_slice=30, figsize=None, legend=True):
    """
    plot multi steps.
    fig_num: Number of pictures drawn, fig_num = -1 drawn all pictures
    """
    assert np.sum([i.ndim == 2 for i in s]) == len(s)
    assert isinstance(fig_num_or_slice, int) or isinstance(fig_num_or_slice, slice)

    if isinstance(fig_num_or_slice, int):
        max_row = np.max([i.shape[0] for i in s])
        if fig_num_or_slice != -1:
            num = fig_num_or_slice if max_row > fig_num_or_slice else max_row
        else:
            num = max_row
        _ = list(s)
    else:
        _ = []
        for i in range(len(s)):
            _.append(s[i][fig_num_or_slice])

        max_row = np.max([i.shape[0] for i in _])
        num = max_row

    cols = 5 if num >= 5 else num
    rows = num // 5 if num % 5 == 0 else num // 5 + 1

    width = 20
    high = rows / cols * 6 if 1 <= rows < 3 else rows * 2.5

    fig_size = figsize if figsize is not None else (width, high)
    plt.figure(figsize=fig_size)

    for j in range(len(_)):
        fig_index = 0
        for i in range(1, num + 1):
            if len(_[j]) > fig_index:
                plt.subplot(rows, cols, i)
                if labels is None:
                    label = f'label_' + str(j + 1)
                else:
                    label = labels[j]
                plt.plot(_[j][fig_index], label=label)
                if legend:
                    plt.legend()
            fig_index += 1
            if fig_index > num:
                break
    plt.show()
