import pandas as pd

from matchify.models.base_model import ERBaseModel
from matchify.utils.data_splitter import DataSplitter


class EntityResolutionPipeline:
    def __init__(self, data_loader, splitter: DataSplitter, model: ERBaseModel):
        self.data_loader = data_loader
        self.splitter = splitter
        self.model = model

    def run(self):
        data = self.data_loader.load()
        self.splitter.data = data
        train_data, test_data, dev_data = self.splitter.split_er()

        self.model.df = train_data
        self.model.train()

        preds = []
        for split in (test_data, dev_data):
            for _, record in split.iterrows():
                preds.append(self.model.predict(record))

        return pd.concat(preds, axis=0) if preds else pd.DataFrame()
