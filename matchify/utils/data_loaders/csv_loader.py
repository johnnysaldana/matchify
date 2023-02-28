import pandas as pd


class CsvLoader:
    def __init__(self, file_path, chunksize=None, low_memory=False):
        """
        Initialize the CsvLoader object.

        Parameters:
            file_path (str): the path to the CSV file.
            chunksize (int): the number of rows to read at a time, or None to read the entire file.
            low_memory (bool): whether to optimize memory usage at the cost of performance.
        """
        self.file_path = file_path
        self.chunksize = chunksize
        self.low_memory = low_memory

    def load(self):
        """
        Load data from a CSV file using pandas.

        Returns:
            pandas DataFrame: the loaded data.
        """
        try:
            if self.chunksize is not None:
                data_chunks = pd.read_csv(
                    self.file_path,
                    chunksize=self.chunksize,
                    low_memory=self.low_memory,
                    iterator=True,
                )
                data = pd.concat(data_chunks, ignore_index=True)
            else:
                data = pd.read_csv(
                    self.file_path,
                    low_memory=self.low_memory,
                )
        except FileNotFoundError:
            raise Exception(f"File {self.file_path} not found.")
        except Exception as e:
            raise Exception(f"Error loading data from {self.file_path}: {e}")

        return data
