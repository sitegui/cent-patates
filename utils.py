import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


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


def load_train_test_data(test_size=0.2, random_state=17):
    """
    Load data and split deterministically into train and test sets
    @returns train_inputs, test_inputs, train_output, test_output
    """
    # Load data and format inputs and outputs
    data = pd.read_csv('data/data.csv')
    inputs_values = [
        data[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']].values,
        data[['lucky_ball']].values,
        data[['wins_1_1_and_0_1']].values
    ]
    output_values = data[[
        'wins_5_1',
        'wins_5_0',
        'wins_4_1_and_4_0',
        'wins_3_1_and_3_0',
        'wins_2_1_and_2_0'
    ]].values

    # Split train/validation
    TEST_SIZE = 0.2
    RANDOM_STATE = 17
    splits = train_test_split(*inputs_values, output_values,
                              test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_inputs = splits[0:6:2]
    test_inputs = splits[1:6:2]
    train_output = splits[6]
    test_output = splits[7]
    return train_inputs, test_inputs, train_output, test_output
