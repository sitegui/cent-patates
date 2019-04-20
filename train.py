import time
from matplotlib.ticker import PercentFormatter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import itertools
from scipy.special import comb

# https://www.tensorflow.org/guide/keras#custom_layers
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer


class FrenchLoterryLayer(tf.keras.layers.Layer):
    """
    A custom-built layer that takes as input a tensor with shape (None,
    6), where the first 5 elements represent the good (winning) normal numbers (from 1
    to 49), in any order, and the last one represents the lucky number (from 1
    to 10).

    The output with shape (None, 5) represents the expected ratio between the
    number of winners in the first 5 levels with the number of winners of the 6h
    level. (See rationale on README)
    """

    def __init__(self, **kwargs):
        # Tell Keras the input shape and type
        kwargs['input_shape'] = (6,)
        kwargs['dtype'] = tf.int32
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        @param input_shape: (TensorShape)
        """
        # Called once from __call__, when we know the shapes of inputs and dtype
        # Should have the calls to add_weight(), and then call the super's build()
        # (which sets self.built = True, which is nice in case the user wants to
        # call build() manually before the first __call__).

        # Ask Keras to create the trainable weights
        self.log_normal_probs = self.add_weight(
            'log_normal_probs', (49,), dtype='float32')
        self.log_lucky_probs = self.add_weight(
            'log_lucky_probs', (10,), dtype='float32')

        # Transform the weights into valid probabilities, so that each element
        # is between 0 and 1 and they all add up to 1
        self.normal_probs = tf.math.softmax(self.log_normal_probs)
        self.lucky_probs = tf.math.softmax(self.log_lucky_probs)

        super().build(input_shape)

    def call(self, inputs):
        """
        @param inputs: (list of Tensor)
        """
        # call(): Called in __call__ after making sure build() has been called once.
        # Should actually perform the logic of applying the layer to the input
        # tensors (which should be passed in as the first argument).

        # Extract the 5 good and lucky numbers from each sample
        good_normal_balls = inputs[:, 0:5]  # shape=(-1, 5)
        good_lucky_ball = inputs[:, 5]  # shape=(-1,)

        # Convert to probability by indexing the layer's weights
        good_normal_probs = tf.gather(self.normal_probs, good_normal_balls-1)
        good_lucky_prob = tf.gather(self.lucky_probs, good_lucky_ball-1)

        # Average prob of bad normal balls
        avg_bad_normal_prob = (
            1 - tf.reduce_sum(good_normal_probs, axis=1)) / 44

        # Compute the probability of each prize level
        prob_5 = self._simple_prob(good_normal_probs, avg_bad_normal_prob, 5)
        prize_1 = prob_5 * good_lucky_prob
        prize_2 = prob_5 * (1 - good_lucky_prob)
        prize_3 = self._simple_prob(good_normal_probs, avg_bad_normal_prob, 4)
        prize_4 = self._simple_prob(good_normal_probs, avg_bad_normal_prob, 3)
        prize_5 = self._simple_prob(good_normal_probs, avg_bad_normal_prob, 2)
        prize_6 = (self._simple_prob(good_normal_probs, avg_bad_normal_prob, 1) +
                   self._simple_prob_zero(avg_bad_normal_prob)) * \
            good_lucky_prob

        # Return a tensor with ratios: prize_i/prize_6 for i = 1,2,3,4,5
        return tf.stack([
            prize_1 / prize_6,
            prize_2 / prize_6,
            prize_3 / prize_6,
            prize_4 / prize_6,
            prize_5 / prize_6
        ], axis=1)

    def compute_output_shape(self, input_shape):
        return (5,)

    def get_normal_probs(self):
        """ Return probabilities for normal numbers as a numpy array with 49 elements """
        return tf.math.softmax(self.log_normal_probs).numpy()

    def get_lucky_probs(self):
        """ Return probabilities for normal numbers as a numpy array with 49 elements """
        return tf.math.softmax(self.log_lucky_probs).numpy()

    def _simple_prob(self, good_probs, avg_bad_prob, num_good):
        """
        @param good_probs: (Tensor float32 (-1, 5))
        @param avg_bad_prob: (Tensor float32 (-1,))
        @para num_good: (int) between 1 and 5 (inclusive)
        @returns (Tensor float32 (-1,))
        """
        with tf.name_scope('take_%d_good' % num_good):
            result = 0
            num_bad = 5 - num_good

            # Sum over all possible combinations of 5 take `num_good`:
            # 1. unstack() will split the initial tensor into 5 columns
            # 2. combinations() will select `num_good` out of those
            # 3. stack() will glue back into a single tensor
            # 4. pad() will fill `num_bad` with the average probability
            good_prob_columns = tf.unstack(good_probs, axis=1)
            for sub_good_prob_columns in itertools.combinations(good_prob_columns, num_good):
                probs = tf.stack(
                    [*sub_good_prob_columns, *([avg_bad_prob] * num_bad)], axis=1)

                # Calculate rho as the average of the probs on each sample
                rho = tf.reduce_mean(probs, axis=1)
                denominator = (1-rho) * (1-2*rho) * (1-3*rho) * (1-4*rho)

                result += 120 * tf.reduce_prod(probs, axis=1) / denominator

            return comb(44, num_bad) * result

    def _simple_prob_zero(self, avg_bad_prob):
        """ Special case of _simple_prob() for num_good=0 """
        rho = avg_bad_prob
        denominator = (1-rho) * (1-2*rho) * (1-3*rho) * (1-4*rho)
        return comb(44, 5) * 120 * tf.pow(avg_bad_prob, 5) / denominator


model = tf.keras.Sequential([FrenchLoterryLayer()])

model.compile(optimizer='adam', loss='mean_squared_error')

data = pd.read_csv('data/data.csv')
input_values = data[[
    'ball_1', 'ball_2', 'ball_3',
    'ball_4', 'ball_5', 'lucky_ball'
]].values
output_values = (data[[
    'wins_5_1', 'wins_5_0',
    'wins_4_1_and_4_0',
    'wins_3_1_and_3_0',
    'wins_2_1_and_2_0'
]]).values / data[['wins_1_1_and_0_1']].values

train_input, test_input, train_output, test_output = \
    train_test_split(input_values, output_values,
                     test_size=0.2, random_state=17)

print('Will train with {} and test with {} samples'.format(
    len(train_input), len(test_input)))

model.fit(
    train_input,
    train_output,
    validation_data=(test_input, test_output),
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.EarlyStopping('loss', patience=5),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/' + time.strftime('%Y%m%d%H%M%S'),
            histogram_freq=1
        )
    ]
)

result = model.layers[0].get_normal_probs()
print(result)
series = pd.Series(result, index=np.arange(1, 50)).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
series.plot.bar(color='C0')
plt.gca().get_yaxis().set_major_formatter(PercentFormatter(1))
plt.grid(axis='y')
plt.xlabel('Number')
plt.ylabel('Probability')
plt.axhline(1/49, color='black', linestyle='--')
plt.ylim(bottom=0.01)
plt.title('Likelihood of each number')
plt.savefig('results/bar.png', bbox_inches='tight')

plt.figure(figsize=(10, 10))
plt.imshow(np.concatenate([result, [np.nan]]).reshape(
    (-1, 5)), cmap=plt.get_cmap('RdYlGn'), vmin=1/49-0.015, vmax=1/49+0.015)
plt.colorbar(format=PercentFormatter(1))
plt.xticks([])
plt.yticks([])
for i in range(49):
    plt.text(i % 5, i//5, str(i+1), horizontalalignment='center',
             verticalalignment='center', fontsize=14)
plt.title('Likelihood of each number')
plt.savefig('results/grid.png', bbox_inches='tight')

np.save('results/result.npy', result)
