import pandas as pd

from matchify.models.entity_resolution_base_model import EntityResolutionBaseModel
from matchify.utils.data_splitter import DataSplitter


class EntityResolutionPipeline:
    def __init__(self, data_loader, splitter, model):
        """
        Initialize the EntityResolutionPipeline object.

        Parameters:
            data_loader (EntityResolutionBaseLoader): the data loader to use.
            splitter (EntityResolver): the data splitter to use.
            model (EntityResolutionBaseModel): the entity resolution model to use.
        """
        self.data_loader = data_loader
        self.splitter = splitter
        self.model = model

    def run(self):
        """
        Run the entity resolution pipeline.

        Returns:
            pandas DataFrame: a DataFrame with columns representing the two records and a match score.
        """
        # Load data
        data = self.data_loader.load()

        # Split data into training, testing, and development datasets
        train_data, test_data, dev_data = self.splitter.split(data)

        # Fit the model to the training data
        self.model.fit(train_data)

        # Predict matches for the testing data
        test_preds = self.model.predict(test_data)

        # Predict matches for the development data
        dev_preds = self.model.predict(dev_data)

        # Combine the testing and development predictions
        preds = pd.concat([test_preds, dev_preds], axis=0)

        return preds
