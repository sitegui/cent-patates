from matplotlib.ticker import PercentFormatter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

# https://www.tensorflow.org/guide/keras#custom_layers
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer


class ExpectationLayer(tf.keras.layers.Layer):
    """
    A custom-built layer that takes as input a tensor with shape (None,
    `num_chosen`+1), where the first `num_chose` elements represent the chosen
    balls, in any order. Valid values are from 1 to `num_total` (inclusive).
    The last element represents the number of games played.

    The output with shape (None, 1) represents the expected number of winners.
    """

    def __init__(self, num_total, num_chosen, **kwargs):
        """
        @param num_total: (int)
        @param num_chosen: (int)
        """
        # Save configuration in member variables
        assert isinstance(num_total, int)
        assert isinstance(num_chosen, int)
        assert num_chosen < num_total
        self.num_total = num_total
        self.num_chosen = num_chosen

        kwargs['input_shape'] = (num_chosen+1,)
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
        print('build')
        assert input_shape.dims[1] == self.num_chosen + 1

        self.log_probs = self.add_weight(
            'log_probs', (self.num_total,), dtype='float32')
        self.probs = tf.math.softmax(self.log_probs)

        super().build(input_shape)

    def call(self, inputs):
        """
        @param inputs: (list of Tensor)
        """
        # call(): Called in __call__ after making sure build() has been called once.
        # Should actually perform the logic of applying the layer to the input
        # tensors (which should be passed in as the first argument).

        print('call')
        gathered_probs = tf.gather(self.probs, inputs[:, :-1] - 1)

        denominator = 1
        avg = tf.math.reduce_mean(gathered_probs, axis=-1)
        for i in range(1, self.num_chosen):
            denominator *= 1 - i * avg

        import math
        expected = tf.cast(inputs[:, -1], 'float32') * \
            math.factorial(self.num_chosen) * \
            tf.math.reduce_prod(gathered_probs, axis=-1) / denominator
        return tf.reshape(expected, (-1, 1))

    def compute_output_shape(self, input_shape):
        print('compute_output_shape')
        return (1,)


model = tf.keras.Sequential()
model.add(ExpectationLayer(49, 5))

model.compile(optimizer='adam', loss='mean_squared_error')
print('done')

data = pd.read_csv('data/data.csv')
data['games'] = data['wins_1_1_and_0_1'] * 10
input_values = data[['ball_1', 'ball_2',
                     'ball_3', 'ball_4', 'ball_5', 'games']].values
output_values = (data['wins_5_1'] + data['wins_5_0']).values

train_input, test_input, train_output, test_output = \
    train_test_split(input_values, output_values,
                     test_size=0.2, random_state=17)

print('Will train with {} and test with {} samples'.format(
    len(train_input), len(test_input)))

print(model.input)
model.fit(train_input, train_output, epochs=1000,
          callbacks=[tf.keras.callbacks.EarlyStopping('loss', patience=5)])

print(model.evaluate(test_input, test_output))

result = tf.math.softmax(model.layers[0].get_weights()).numpy().flatten()
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
plt.savefig('bar.png', bbox_inches='tight')

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
plt.savefig('grid.png', bbox_inches='tight')
