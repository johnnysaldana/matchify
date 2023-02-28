import pandas as pd
import boto3


class S3Loader:
    def __init__(self, bucket, key, chunksize=None, region=None, profile=None, **kwargs):
        """
        Initialize the S3Loader object.

        Parameters:
            bucket (str): the name of the S3 bucket containing the object.
            key (str): the key of the object to read.
            chunksize (int): the number of rows to read at a time, or None to read the entire file.
            region (str): the name of the region containing the S3 bucket.
            profile (str): the name of the AWS profile to use for authentication.
            **kwargs: additional arguments to pass to the S3 resource or client.
        """
        self.bucket = bucket
        self.key = key
        self.chunksize = chunksize
        self.region = region
        self.profile = profile
        self.kwargs = kwargs

    def load(self):
        """
        Load data from an object in an S3 bucket using boto3 and pandas.

        Returns:
            pandas DataFrame: the loaded data.
        """
        try:
            session = boto3.Session(profile_name=self.profile, region_name=self.region)
            s3 = session.resource('s3', **self.kwargs)
            obj = s3.Object(self.bucket, self.key)
            if self.chunksize is not None:
                data_chunks = pd.read_csv(obj.get()['Body'], chunksize=self.chunksize)
                data = pd.concat(data_chunks, ignore_index=True)
            else:
                data = pd.read_csv(obj.get()['Body'])
        except Exception as e:
            raise Exception(f"Error loading data from S3 bucket: {e}")

        return data
