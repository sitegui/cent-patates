import math
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from cent_patates.model_data import ModelData


def plot_probs_in_bars(series, color, ax=None, title='Probability of choosing a given number'):
    if ax:
        plt.sca(ax)
    series.plot.bar(color=color)
    plt.axhline(1 / len(series), color='black', linestyle='--')
    plt.gca().get_yaxis().set_major_formatter(PercentFormatter(1))
    plt.title(title)
    plt.legend().set_visible(False)


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
        plt.figure(figsize=(int(grid_width * 1.5), int(grid_height * 1.25)))

    plt.imshow(grid, norm=norm)
    plt.colorbar(format=PercentFormatter(1),
                 orientation='horizontal' if grid_width > grid_height else 'vertical')
    plt.xticks([])
    plt.yticks([])
    for i, index in enumerate(series.index):
        plt.text(i % grid_width,
                 i // grid_width,
                 str(index),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.title(title)


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 14, 'figure.figsize': [10, 6]})

    normal_probs = pd.read_csv('results/normal_probs.csv', index_col=0)['prob']
    lucky_probs = pd.read_csv('results/lucky_probs.csv', index_col=0)['prob']

    plot_probs_in_bars(normal_probs, 'C0')
    plt.savefig('results/normal_probs_bar.png')
    plt.clf()

    plot_probs_in_bars(lucky_probs, 'C1')
    plt.savefig('results/lucky_probs_bar.png')
    plt.clf()

    epoch_normal_probs = pd.read_csv('results/epoch_normal_probs.csv', index_col=0)
    colors = [f'C{i // 10}' for i in range(49)]
    epoch_normal_probs.plot.line(xlabel='Epoch', ylabel='Likelihood', legend=None, color=colors)
    plt.gca().get_yaxis().set_major_formatter(PercentFormatter(1))
    legend_handles = []
    for i in range(0, 49, 10):
        legend_handles.append(plt.plot([], label=f'{i + 1} - {i + 10}', color=f'C{i // 10}')[0])
    plt.legend(handles=legend_handles, loc='lower center', ncols=5)
    plt.title('How likelihood evolves during training for each number')
    plt.savefig('results/epoch_normal_probs.png')
    plt.clf()

    plot_probs_in_grid(normal_probs)
    plt.savefig('results/normal_probs_grid.png')
    plt.clf()
    plot_probs_in_grid(lucky_probs)
    plt.savefig('results/lucky_probs_grid.png')
    plt.clf()

    top_5 = normal_probs.nlargest(5)
    bottom_5 = normal_probs.nsmallest(5)
    print('Most likely play: ' + str(top_5.index.values) + ' ' + str(lucky_probs.idxmax()))
    print('Least likely play: ' + str(bottom_5.index.values) + ' ' + str(lucky_probs.idxmin()))
    gap = np.prod(top_5) * lucky_probs.max() / np.prod(bottom_5) / lucky_probs.min()
    print(f'People prefer the first one {gap:.1f}x more')

    model_data = ModelData()

    normal_frequency = Counter({n: 0 for n in range(1, 50)})
    lucky_frequency = Counter({n: 0 for n in range(1, 11)})
    for i in range(1, 6):
        normal_frequency.update(model_data.full_df[f'ball_{i}'].values)
    lucky_frequency.update(model_data.full_df['lucky_ball'].values)
    normal_frequency = pd.Series(normal_frequency)
    normal_frequency /= normal_frequency.sum()
    lucky_frequency = pd.Series(lucky_frequency)
    lucky_frequency /= lucky_frequency.sum()

    top_5 = normal_frequency.nlargest(5)
    bottom_5 = normal_frequency.nsmallest(5)
    print('Most frequent draw: ' + str(top_5.index.values) + ' ' + str(lucky_frequency.idxmax()))
    print('Least frequent draw: ' + str(bottom_5.index.values) + ' ' +
          str(lucky_frequency.idxmin()))
    gap = np.prod(top_5) * lucky_frequency.max() / np.prod(bottom_5) / lucky_frequency.min()
    print(f'First one {gap:.1f}x more frequent')

    fig, axs = plt.subplots(1, 2)
    norm = Normalize(min(normal_frequency.min(), normal_probs.min()),
                     max(normal_frequency.max(), normal_probs.max()))
    plot_probs_in_grid(normal_frequency, ax=axs[0], title='Draw frequency', norm=norm)
    plot_probs_in_grid(normal_probs, ax=axs[1], title='Players\' bias', norm=norm)
    plt.savefig('results/normal_player_vs_draw.png')
    plt.clf()

    fig, axs = plt.subplots(1, 2)
    norm = Normalize(min(lucky_frequency.min(), lucky_probs.min()),
                     max(lucky_frequency.max(), lucky_probs.max()))
    plot_probs_in_grid(lucky_frequency, ax=axs[0], title='Draw frequency', norm=norm)
    plot_probs_in_grid(lucky_probs, ax=axs[1], title='Players\' bias', norm=norm)
    plt.savefig('results/lucky_player_vs_draw.png')
    plt.clf()

    predicted = pd.read_csv('results/predicted.csv').merge(model_data.full_df, on='date')
    predicted['weekday'] = pd.to_datetime(predicted['date']).dt.weekday.map({
        0: 'Monday',
        2: 'Wednesday',
        5: 'Saturday'
    })

    plt.figure()
    sns.histplot(predicted['predicted_players'])
    plt.title('Distribution of expected number of players')
    plt.savefig('results/player_distribution.png')
    plt.clf()

    plt.figure()
    for wd in predicted['weekday'].unique():
        df = predicted[predicted['weekday'] == wd]
        sns.histplot(df['predicted_players'], label=wd)
    plt.legend()
    plt.title('Do people play more on a given day?')
    plt.savefig('results/player_distribution_by_day.png')
    plt.clf()

    sns.lmplot(
        data=predicted[predicted['jackpot'] != 0],
        x='jackpot',
        y='predicted_players',
        hue='weekday',
        aspect=2,
    )
    plt.title('Do more people play the bigger the Jackpot?')
    plt.savefig('results/player_distribution_by_jackpot.png')
    plt.clf()

    # Produce detailed per-epoch charts
    os.makedirs('results/epoch', exist_ok=True)
    epoch_normal_probs = pd.read_csv('results/epoch_normal_probs.csv', index_col=0)
    for i in range(len(epoch_normal_probs.index)):
        plot_probs_in_bars(epoch_normal_probs.iloc[i], 'C0')
        plt.savefig(f'results/epoch/normal_probs_{i:02}.png')
        plt.clf()

    epoch_lucky_probs = pd.read_csv('results/epoch_lucky_probs.csv', index_col=0)
    for i in range(len(epoch_lucky_probs.index)):
        plot_probs_in_bars(epoch_lucky_probs.iloc[i], 'C1')
        plt.savefig(f'results/epoch/lucky_probs_{i:02}.png')
        plt.clf()
