import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot_probs_in_bars(series, color, ax=None, title='Probability of choosing a given number'):
    if ax:
        plt.sca(ax)
    series.sort_values(ascending=False).plot.bar(
        figsize=(14, 5), color=color, width=0.5)
    plt.axhline(1 / len(series), color='black', linestyle='--')
    plt.gca().get_yaxis().set_major_formatter(PercentFormatter(1))
    plt.title(title)


def plot_probs_in_grid(series, ax=None, title='Likelihood of each number', norm=None):
    grid_width = 5
    grid_height = int(math.ceil(len(series) / grid_width))
    grid = np.full(grid_height * grid_width, np.nan)
    series = series.sort_index()
    grid[:len(series)] = series.values
    grid = np.reshape(grid, (grid_height, grid_width))

    if ax:
        plt.sca(ax)
    else:
        plt.figure(figsize=(int(grid_width*1.5), int(grid_height*1.25)))

    plt.imshow(grid, cmap=plt.get_cmap('RdYlGn'), norm=norm)
    plt.colorbar(format=PercentFormatter(
        1), orientation='horizontal' if grid_width > grid_height else 'vertical')
    plt.xticks([])
    plt.yticks([])
    for i, index in enumerate(series.index):
        plt.text(i % grid_width, i//grid_width, str(index), horizontalalignment='center',
                 verticalalignment='center')
    plt.title(title)
