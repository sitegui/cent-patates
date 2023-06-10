import pandas as pd
from sklearn.model_selection import train_test_split


class ModelData:
    """ Represent data to interact with the model """

    def __init__(self, test_size: float = 0.2, random_state: int = 17, num_examples: int = 3):
        self.full_df = pd.read_csv('data/data.csv')
        self.inputs = [
            self.full_df[['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']].values,
            self.full_df[['lucky_ball']].values,
            self.full_df[['wins_1_1_and_0_1']].values,
        ]
        self.output = self.full_df[[
            'wins_5_1', 'wins_5_0', 'wins_4_1_and_4_0', 'wins_3_1_and_3_0', 'wins_2_1_and_2_0'
        ]].values

        # Split train/validation
        splits = train_test_split(
            *self.inputs,
            self.output,
            self.full_df['date'].values,
            test_size=test_size,
            random_state=random_state,
        )
        self.train_inputs = splits[0:6:2]
        self.test_inputs = splits[1:6:2]
        self.train_output = splits[6]
        self.test_output = splits[7]
        test_dates = splits[9]

        # Take a set of examples
        self.num_examples = num_examples
        self.example_dates = test_dates[:num_examples]
        self.example_inputs = [input[:num_examples, :] for input in self.test_inputs]
        self.example_outputs = self.test_output[:num_examples, :]
