import numpy as np
import pandas as pd
import tensorflow as tf

from cent_patates.model import build_model
from cent_patates.model_data import ModelData


class AccumulateEpochProbs(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self._epoch_normal_probs = []
        self._epoch_lucky_probs = []
        self._epoch_prize_probs = []
        self._epoch_prize_wins = []
        self._extracting_model = None

    def on_train_begin(self, logs=None):
        self._extracting_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer('calculate_prize_probs').output,
                self.model.outputs[1],
            ],
        )

    def on_epoch_begin(self, epoch, logs=None):
        self._accumulate()

    def on_train_end(self, logs=None):
        self._accumulate()

    def _accumulate(self):
        normal_probs, lucky_probs = self.model.get_layer('gather_probs_layer').get_probs()
        self._epoch_normal_probs.append(normal_probs)
        self._epoch_lucky_probs.append(lucky_probs)
        prize_probs, prize_wins = self._extracting_model(model_data.example_inputs)
        self._epoch_prize_probs.append(prize_probs)
        self._epoch_prize_wins.append(prize_wins)

    def epoch_probs(self):
        num_epochs = len(self._epoch_prize_probs)
        epoch_prize_dates = np.tile(model_data.example_dates, num_epochs)

        epoch_prize_probs = pd.DataFrame(
            np.concatenate(self._epoch_prize_probs, axis=0),
            columns=[
                'prob_5_1',
                'prob_5_0',
                'prob_4_1_and_4_0',
                'prob_3_1_and_3_0',
                'prob_2_1_and_2_0',
                'prob_1_1_and_0_1',
            ],
        )
        epoch_prize_probs.insert(0, 'date', epoch_prize_dates)
        epoch_prize_probs.insert(0, 'epoch',
                                 epoch_prize_probs.index.values // model_data.num_examples)

        epoch_prize_wins = pd.DataFrame(
            np.concatenate(self._epoch_prize_wins, axis=0),
            columns=[
                'wins_5_1',
                'wins_5_0',
                'wins_4_1_and_4_0',
                'wins_3_1_and_3_0',
                'wins_2_1_and_2_0',
            ],
        )
        epoch_prize_wins.insert(0, 'date', epoch_prize_dates)
        epoch_prize_wins.insert(0, 'epoch',
                                epoch_prize_wins.index.values // model_data.num_examples)

        return (
            pd.DataFrame(np.stack(self._epoch_normal_probs), columns=np.arange(1, 50)),
            pd.DataFrame(np.stack(self._epoch_lucky_probs), columns=np.arange(1, 11)),
            epoch_prize_probs,
            epoch_prize_wins,
        )


if __name__ == '__main__':
    model_data = ModelData()

    print(
        f'Will train with {len(model_data.train_inputs[0])} and test with {len(model_data.test_inputs[0])} samples'
    )

    avg_wins = np.mean(model_data.train_output, axis=0)

    def custom_loss(y_true, y_pred):
        normalized_error = (y_pred - tf.cast(y_true, 'float32')) / avg_wins
        return tf.reduce_mean(tf.math.square(normalized_error), axis=1)

    accumulate_epoch_probs = AccumulateEpochProbs()

    model = build_model()
    model.compile(optimizer='adam', loss=[None, custom_loss])
    model.fit(
        model_data.train_inputs,
        model_data.train_output,
        validation_data=(model_data.test_inputs, model_data.test_output),
        epochs=1000,
        callbacks=[
            tf.keras.callbacks.EarlyStopping('loss', patience=10),
            tf.keras.callbacks.TensorBoard(),
            accumulate_epoch_probs,
        ],
    )

    normal_probs, lucky_probs = model.get_layer('gather_probs_layer').get_probs()
    normal_probs = pd.Series(normal_probs, index=np.arange(1, 50), name='prob')
    lucky_probs = pd.Series(lucky_probs, index=np.arange(1, 11), name='prob')
    normal_probs.to_csv('results/normal_probs.csv')
    lucky_probs.to_csv('results/lucky_probs.csv')

    epoch_normal_probs, epoch_lucky_probs, epoch_prize_probs, epoch_prize_wins = accumulate_epoch_probs.epoch_probs(
    )
    epoch_normal_probs.to_csv('results/epoch_normal_probs.csv', index_label='epoch')
    epoch_lucky_probs.to_csv('results/epoch_lucky_probs.csv', index_label='epoch')
    epoch_prize_probs.to_csv('results/epoch_prize_probs.csv', index=False)
    epoch_prize_wins.to_csv('results/epoch_prize_wins.csv', index=False)

    predicted_players, predicted_wins = model.predict(model_data.inputs)
    pd.DataFrame({
        'date': model_data.full_df['date'],
        'predicted_players': predicted_players[:, 0],
        'predicted_wins_5_1': predicted_wins[:, 0],
        'predicted_wins_5_0': predicted_wins[:, 1],
        'predicted_wins_4_1_and_4_0': predicted_wins[:, 2],
        'predicted_wins_3_1_and_3_0': predicted_wins[:, 3],
        'predicted_wins_2_1_and_2_0': predicted_wins[:, 4],
    }).to_csv('results/predicted.csv', index=False)
