import pandas as pd
import pyarrow.parquet as pq


class ParquetLoader:
    def __init__(self, file_path, columns=None, chunksize=None):
        """
        Initialize the ParquetLoader object.

        Parameters:
            file_path (str): the path to the Parquet file.
            columns (list): the list of columns to read, or None to read all columns.
            chunksize (int): the number of records to read at a time, or None to read the entire file.
        """
        self.file_path = file_path
        self.columns = columns
        self.chunksize = chunksize

    def load(self):
        """
        Load data from a Parquet file using pyarrow and pandas.

        Returns:
            pandas DataFrame: the loaded data.
        """
        try:
            if self.chunksize is not None:
                data_chunks = []
                with pq.ParquetFile(self.file_path) as pf:
                    num_rows = pf.metadata.num_rows
                    for i in range(0, num_rows, self.chunksize):
                        table = pf.read_row_group(i, self.chunksize, columns=self.columns)
                        data_chunks.append(table.to_pandas())
                        if i % (self.chunksize * 10) == 0:
                            yield pd.concat(data_chunks, ignore_index=True)
                            data_chunks = []
                    if data_chunks:
                        yield pd.concat(data_chunks, ignore_index=True)
                data = pd.concat(data_chunks, ignore_index=True)
            else:
                table = pq.read_table(self.file_path, columns=self.columns)
                data = table.to_pandas()
        except FileNotFoundError:
            raise Exception(f"File {self.file_path} not found.")
        except Exception as e:
            raise Exception(f"Error loading data from {self.file_path}: {e}")

        return data
