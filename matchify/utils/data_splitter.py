from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, data, target_col, test_size=0.2, dev_size=0.1, random_state=None):
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.dev_size = dev_size
        self.random_state = random_state

    def split(self):
        train_data, test_data = train_test_split(
            self.data,
            test_size=self.test_size,
            stratify=self.data[self.target_col],
            random_state=self.random_state,
        )
        dev_data, test_data = train_test_split(
            test_data,
            test_size=self.dev_size / (1 - self.test_size),
            stratify=test_data[self.target_col],
            random_state=self.random_state,
        )
        return train_data, test_data, dev_data

    def split_er(self):
        # split by group_id so a duplicate group never crosses partitions
        unique_ids = self.data[self.target_col].unique()
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=self.test_size,
            random_state=self.random_state
        )
        dev_ids, test_ids = train_test_split(
            test_ids,
            test_size=self.dev_size / (1 - self.test_size),
            random_state=self.random_state
        )
        train_data = self.data[self.data[self.target_col].isin(train_ids)]
        test_data = self.data[self.data[self.target_col].isin(test_ids)]
        dev_data = self.data[self.data[self.target_col].isin(dev_ids)]
        return train_data, test_data, dev_data
