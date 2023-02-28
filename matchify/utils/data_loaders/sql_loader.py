import pandas as pd
import sqlalchemy


class SqlLoader:
    def __init__(self, connection_string, query, chunksize=None, schema=None, table=None, params=None):
        """
        Initialize the SqlLoader object.

        Parameters:
            connection_string (str): the connection string for the SQL database.
            query (str): the SQL query to execute.
            chunksize (int): the number of rows to read at a time, or None to read the entire result set.
            schema (str): the name of the schema containing the table.
            table (str): the name of the table to read.
            params (dict): a dictionary of parameters to pass to the query.
        """
        self.connection_string = connection_string
        self.query = query
        self.chunksize = chunksize
        self.schema = schema
        self.table = table
        self.params = params or {}

    def load(self):
        """
        Load data from a SQL database using sqlalchemy.

        Returns:
            pandas DataFrame: the loaded data.
        """
        try:
            engine = sqlalchemy.create_engine(self.connection_string)
            with engine.connect() as conn:
                query = sqlalchemy.text(self.query)
                if self.chunksize is not None:
                    data_chunks = pd.read_sql(
                        query,
                        conn,
                        params=self.params,
                        chunksize=self.chunksize,
                        schema=self.schema,
                        table=self.table,
                    )
                    data = pd.concat(data_chunks, ignore_index=True)
                else:
                    data = pd.read_sql(
                        query,
                        conn,
                        params=self.params,
                        schema=self.schema,
                        table=self.table,
                    )
        except Exception as e:
            raise Exception(f"Error loading data from SQL database: {e}")

        return data
