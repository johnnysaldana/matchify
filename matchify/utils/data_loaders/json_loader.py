import json
import pandas as pd


class JsonLoader:
    def __init__(self, file_path, lines=False, chunksize=None):
        """
        Initialize the JsonLoader object.

        Parameters:
            file_path (str): the path to the JSON file.
            lines (bool): whether the JSON file contains one JSON object per line.
            chunksize (int): the number of records to read at a time, or None to read the entire file.
        """
        self.file_path = file_path
        self.lines = lines
        self.chunksize = chunksize

    def load(self):
        """
        Load data from a JSON file using json and pandas.

        Returns:
            pandas DataFrame: the loaded data.
        """
        try:
            if self.chunksize is not None:
                with open(self.file_path, 'r') as f:
                    data_chunks = []
                    for i, line in enumerate(f):
                        if self.lines:
                            obj = json.loads(line)
                        else:
                            obj = json.loads(line.strip('[]\n,'))
                        data_chunks.append(obj)
                        if i % self.chunksize == self.chunksize - 1:
                            data = pd.DataFrame(data_chunks)
                            yield data
                            data_chunks = []
                    if data_chunks:
                        data = pd.DataFrame(data_chunks)
                        yield data
                data = pd.concat(data_chunks, ignore_index=True)
            else:
                if self.lines:
                    data = pd.read_json(self.file_path, lines=True)
                else:
                    data = pd.read_json(self.file_path)
        except FileNotFoundError:
            raise Exception(f"File {self.file_path} not found.")
        except Exception as e:
            raise Exception(f"Error loading data from {self.file_path}: {e}")

        return data
