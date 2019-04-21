import tensorflow as tf
import itertools
from scipy.special import comb

# https://www.tensorflow.org/guide/keras#custom_layers
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer


class GatherProbsLayer(tf.keras.layers.Layer):
    """
    A custom-built layer to map ball numbers into their individual probabilities

    Input: [int32 (None, 5), int32 (None, 1)]
    The first input has 5 elements representing the good (winning) normal numbers (from 1
    to 49), in any order.
    The second input represents the lucky number (from 1 to 10).

    Output: [float32 (None, 5), float32 (None, 1)] represents each individual probability
    """

    def build(self, input_shape):
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
        # call(): Called in __call__ after making sure build() has been called once.
        # Should actually perform the logic of applying the layer to the input
        # tensors (which should be passed in as the first argument).

        # Convert to probability by indexing the layer's weights
        good_normal_probs = tf.gather(self.normal_probs, inputs[0]-1)
        good_lucky_prob = tf.gather(self.lucky_probs, inputs[1]-1)
        return [good_normal_probs, good_lucky_prob]

    def get_probs(self):
        """ Return probabilities for normal and lucky numbers as two numpy arrays """
        return tf.math.softmax(self.log_normal_probs).numpy(), tf.math.softmax(self.log_lucky_probs).numpy()


class CalculatePrizeProbs(tf.keras.layers.Layer):
    """
    A layer that calculates the probability of players winning each
    one of the 6 prize categories.
    Input: [float32 (None, 5), float32 (None, 1)]
    Output: float32 (None, 6)
    """

    def call(self, inputs):
        # Extract the 5 good and lucky probs from each sample
        good_normal_probs = inputs[0]
        good_lucky_prob = tf.reshape(inputs[1], (-1,))

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

        return tf.stack([prize_1, prize_2, prize_3, prize_4, prize_5, prize_6], axis=1)

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
            good_prob_columns = tf.unstack(good_probs, num=5, axis=1)
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


class CalculateExpectedPlayers(tf.keras.layers.Layer):
    """
    A layer that takes as input the number of winners of the 6th prize and the
    probabilities of each prize and outputs the expected number of players and
    winners in the other 5 categories
    Input: [int32 (None, 1), float32 (None, 6)]
    Output: [float32 (None, 1), float32 (None, 5)]
    """

    def call(self, inputs):
        winners_6 = tf.cast(inputs[0], 'float32')
        prize_probs = inputs[1]
        expected_players = winners_6 / prize_probs[:, 5:6]
        expected_winners = prize_probs[:, 0:5] * expected_players
        return [expected_players, expected_winners]


# Build the model: define inputs
good_numbers = tf.keras.layers.Input(shape=(5,), dtype='int32')
lucky_number = tf.keras.layers.Input(shape=(1,), dtype='int32')
prize_6_winners = tf.keras.layers.Input(shape=(1,), dtype='int32')

# Build the model: define layers
good_probs, lucky_prob = GatherProbsLayer()([good_numbers, lucky_number])
prize_probs = CalculatePrizeProbs()([good_probs, lucky_prob])
players, prize_winners = CalculateExpectedPlayers()(
    [prize_6_winners, prize_probs])

# Build the model
model = tf.keras.Model(inputs=[
    good_numbers,
    lucky_number,
    prize_6_winners
], outputs=[
    players,
    prize_winners
])
