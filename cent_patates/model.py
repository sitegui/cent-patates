from typing import Tuple

import numpy as np
import tensorflow as tf

from cent_patates.prob_fns import (prob_1_1_and_0_1, prob_2_1_and_2_0, prob_3_1_and_3_0,
                                   prob_4_1_and_4_0, prob_5_0, prob_5_1)


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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_normal_probs = self.add_weight('log_normal_probs', (49,), dtype='float32')
        self.log_lucky_probs = self.add_weight('log_lucky_probs', (10,), dtype='float32')

    def call(self, inputs):
        # Transform the weights into valid probabilities, so that each element
        # is between 0 and 1 and they all add up to 1
        normal_probs = tf.math.softmax(self.log_normal_probs)
        lucky_probs = tf.math.softmax(self.log_lucky_probs)

        # Convert to probability by indexing the layer's weights
        good_normal_probs = tf.gather(normal_probs, inputs[0] - 1)
        good_lucky_prob = tf.gather(lucky_probs, inputs[1] - 1)
        return [good_normal_probs, good_lucky_prob]

    def get_probs(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return probabilities for normal and lucky numbers as two numpy arrays """
        return tf.math.softmax(self.log_normal_probs).numpy(), tf.math.softmax(
            self.log_lucky_probs).numpy()


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
        good_lucky_prob = inputs[1]

        # Compute the probability of each prize level
        return tf.concat([
            prob_5_1(good_normal_probs, good_lucky_prob),
            prob_5_0(good_normal_probs, good_lucky_prob),
            prob_4_1_and_4_0(good_normal_probs),
            prob_3_1_and_3_0(good_normal_probs),
            prob_2_1_and_2_0(good_normal_probs),
            prob_1_1_and_0_1(good_normal_probs, good_lucky_prob)
        ],
            axis=1)


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


def build_model() -> tf.keras.Model:
    # Build the model: define inputs
    good_numbers = tf.keras.layers.Input(shape=(5,), dtype='int32')
    lucky_number = tf.keras.layers.Input(shape=(1,), dtype='int32')
    prize_6_winners = tf.keras.layers.Input(shape=(1,), dtype='int32')

    # Build the model: define layers
    good_probs, lucky_prob = GatherProbsLayer()([good_numbers, lucky_number])
    prize_probs = CalculatePrizeProbs()([good_probs, lucky_prob])
    players, prize_winners = CalculateExpectedPlayers()([prize_6_winners, prize_probs])

    # Build the model
    return tf.keras.Model(
        inputs=[good_numbers, lucky_number, prize_6_winners],
        outputs=[players, prize_winners],
    )
