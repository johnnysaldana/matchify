import pandas as pd
import numpy as np
from itertools import combinations

from matchify.models.entity_resolution_base_model import EntityResolutionBaseModel


class ExactMatchModel(EntityResolutionBaseModel):
    def __init__(self, **kwargs):
        """
        Initialize the ExactMatchModel object.

        Parameters:
            **kwargs: any additional arguments to be passed to the model.
        """
        super().__init__(**kwargs)
        self.groups = None

    def fit(self, dataframes):
        """
        Find exact matches between records in the input dataframes.

        Parameters:
            dataframes (list): a list of pandas DataFrames to find matches in.

        Returns:
            None
        """
        # Concatenate dataframes into a single dataframe
        data = pd.concat(dataframes, axis=0)

        # Get unique records and assign group numbers
        unique_records, self.groups = self._get_unique_records(data)

        # Create a dictionary mapping records to group numbers
        self.group_dict = dict(zip(unique_records, self.groups))

    def predict(self, data):
        """
        Predict the matches for the input data using the group numbers from the fit method.

        Parameters:
            data (pandas DataFrame): the input data to predict matches for.

        Returns:
            pandas DataFrame: a DataFrame with columns representing the two records and a match score.
        """
        # Get unique records in the input data
        unique_records, _ = self._get_unique_records(data)

        # Get group numbers for each record using the dictionary created in the fit method
        group_nums = [self.group_dict.get(record, np.nan) for record in unique_records]

        # Add group numbers as a column to the input data and return
        output_data = data.copy()
        output_data['group'] = group_nums
        return output_data

    def _get_unique_records(self, data):
        """
        Find unique records in the input data and assign group numbers.

        Parameters:
            data (pandas DataFrame): the input data to find unique records in.

        Returns:
            Tuple[pandas DataFrame, numpy array]: the unique records and their group numbers.
        """
        # Get unique records and assign group numbers
        unique_records = data.drop_duplicates(keep='first')
        groups = np.arange(1, len(unique_records) + 1)

        # Create a dictionary mapping record tuples to group numbers
        record_dict = dict(zip(map(tuple, unique_records.to_numpy()), groups))

        # Iterate through all pairs of records and assign the same group number if they match
        for r1, r2 in combinations(unique_records.to_numpy(), 2):
            if np.all(r1 == r2):
                group_num = record_dict[tuple(r1)]
                record_dict[tuple(r2)] = group_num

        # Get final groups for each unique record
        final_groups = np.array([record_dict[tuple(r)] for r in unique_records.to_numpy()])

        return unique_records, final_groups
