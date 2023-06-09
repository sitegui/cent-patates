import time
import numpy as np
import pandas as pd
import tensorflow as tf

from model import model
from utils import load_train_test_data

train_inputs, test_inputs, train_output, test_output = load_train_test_data()

print('Will train with {} and test with {} samples'.format(
    len(train_inputs[0]), len(test_inputs[0])))

avg_winners = np.mean(train_output, axis=0)


def custom_loss(y_true, y_pred):
    normalized_error = (y_pred - tf.cast(y_true, 'float32')) / avg_winners
    return tf.reduce_mean(tf.math.square(normalized_error), axis=1)


class AccumulateEpochProbs(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self._epoch_normal_probs = []
        self._epoch_lucky_probs = []

    def on_epoch_begin(self, epoch, logs=None):
        self._accumulate()

    def on_train_end(self, logs=None):
        self._accumulate()

    def _accumulate(self):
        normal_probs, lucky_probs = self.model.get_layer('gather_probs_layer').get_probs()
        self._epoch_normal_probs.append(normal_probs)
        self._epoch_lucky_probs.append(lucky_probs)

    def epoch_probs(self):
        return np.stack(self._epoch_normal_probs), np.stack(self._epoch_lucky_probs)


accumulate_epoch_probs = AccumulateEpochProbs()

model.compile(optimizer='adam', loss=[None, custom_loss])
model.fit(
    train_inputs,
    train_output,
    validation_data=(test_inputs, test_output),
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.EarlyStopping('loss', patience=5),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/' + time.strftime('%Y-%m-%d %H-%M-%S'),
        ),
        accumulate_epoch_probs,
    ]
)

model.save('results/model.h5', include_optimizer=False)
normal_probs, lucky_probs = model.get_layer('gather_probs_layer').get_probs()
normal_probs = pd.Series(normal_probs, index=np.arange(
    1, 50))
lucky_probs = pd.Series(lucky_probs, index=np.arange(
    1, 11))
normal_probs.to_csv('results/normal_probs.csv', header=False)
lucky_probs.to_csv('results/lucky_probs.csv', header=False)

epoch_normal_probs, epoch_lucky_probs = accumulate_epoch_probs.epoch_probs()
pd.DataFrame(epoch_normal_probs).to_csv('results/epoch_normal_probs.csv', header=False)
pd.DataFrame(epoch_lucky_probs).to_csv('results/epoch_lucky_probs.csv', header=False)
