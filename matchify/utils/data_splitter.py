import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, data, target_col, test_size=0.2, dev_size=0.1, random_state=None):
        """
        Initialize the DataSplitter object.

        Parameters:
            data (pandas DataFrame): the input data to split.
            target_col (str): the name of the target column.
            test_size (float): the proportion of data to use for testing.
            dev_size (float): the proportion of data to use for development.
            random_state (int): the random seed to use for the split.
        """
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.dev_size = dev_size
        self.random_state = random_state

    def split(self):
        """
        Split the input data into training, testing, and development datasets.

        Returns:
            Tuple[pandas DataFrame, pandas DataFrame, pandas DataFrame]: the training, testing, and development datasets.
        """
        # Split data into training and testing datasets
        train_data, test_data = train_test_split(
            self.data,
            test_size=self.test_size,
            stratify=self.data[self.target_col],
            random_state=self.random_state,
        )

        # Split remaining data into development and testing datasets
        dev_data, test_data = train_test_split(
            test_data,
            test_size=self.dev_size / (1 - self.test_size),
            stratify=test_data[self.target_col],
            random_state=self.random_state,
        )

        return train_data, test_data, dev_data
