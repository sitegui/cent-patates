import time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

from model import model

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

print('Will train with {} and test with {} samples'.format(
    len(train_inputs[0]), len(test_inputs[0])))

avg_winners = np.mean(output_values, axis=0)


def custom_loss(y_true, y_pred):
    normalized_error = (y_pred - y_true) / avg_winners
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
            log_dir='logs/' + time.strftime('%Y%m%d%H%M%S'),
            histogram_freq=1
        )
    ]
)

model.save('results/model.h5', include_optimizer=False)
