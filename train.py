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
            histogram_freq=1
        )
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
