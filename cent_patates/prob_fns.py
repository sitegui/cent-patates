# Implement the main probability functions, used by the model

import itertools
from math import comb

import tensorflow as tf


def prob_5_1(good_normal_probs: tf.Tensor, good_lucky_prob: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_5_1'):
        return _simple_prob(good_normal_probs, 5) * good_lucky_prob


def prob_5_0(good_normal_probs: tf.Tensor, good_lucky_prob: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_5_0'):
        return _simple_prob(good_normal_probs, 5) * (1 - good_lucky_prob)


def prob_4_1_and_4_0(good_normal_probs: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_4_1_and_4_0'):
        return _simple_prob(good_normal_probs, 4)


def prob_3_1_and_3_0(good_normal_probs: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_3_1_and_3_0'):
        return _simple_prob(good_normal_probs, 3)


def prob_2_1_and_2_0(good_normal_probs: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_2_1_and_2_0'):
        return _simple_prob(good_normal_probs, 2)


def prob_1_1_and_0_1(good_normal_probs: tf.Tensor, good_lucky_prob: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('prob_1_1_and_0_1'):
        prob1 = _simple_prob(good_normal_probs, 1)
        prob0 = _simple_prob(good_normal_probs, 0)
        return (prob1 + prob0) * good_lucky_prob


def _simple_prob(good_probs: tf.Tensor, num_good: int) -> tf.Tensor:
    """
    @param good_probs: (Tensor float32 (-1, 5))
    @para num_good: (int) between 0 and 5 (inclusive)
    @returns (Tensor float32 (-1, 1))
    """
    # Average prob of bad normal balls
    avg_bad_prob = (1 - tf.reduce_sum(good_probs, axis=1)) / 44

    if num_good == 0:
        # Special case of _simple_prob() for num_good=0
        rho = avg_bad_prob
        denominator = (1 - rho) * (1 - 2 * rho) * (1 - 3 * rho) * (1 - 4 * rho)
        return tf.reshape(comb(44, 5) * 120 * tf.pow(avg_bad_prob, 5) / denominator, (-1, 1))

    result = 0
    num_bad = 5 - num_good

    # Sum over all possible combinations of 5 take `num_good`:
    # 1. unstack() will split the initial tensor into 5 columns
    # 2. combinations() will select `num_good` out of those
    # 3. stack() will glue back into a single tensor
    good_prob_columns = tf.unstack(good_probs, num=5, axis=1)
    for sub_good_prob_columns in itertools.combinations(good_prob_columns, num_good):
        probs = tf.stack([*sub_good_prob_columns, *([avg_bad_prob] * num_bad)], axis=1)

        # Calculate rho as the average of the probs on each sample
        rho = tf.reduce_mean(probs, axis=1)
        denominator = (1 - rho) * (1 - 2 * rho) * (1 - 3 * rho) * (1 - 4 * rho)

        result += 120 * \
                  tf.reduce_prod(probs, axis=1) / denominator

    return tf.reshape(comb(44, num_bad) * result, (-1, 1))
