import pandas as pd
from matchify.models.base_model import ERBaseModel


class ExactMatchModel(ERBaseModel):
    def preprocess(self) -> pd.DataFrame:
        """
        Since the exact matching doesn't require any specific preprocessing, we just return
        the original data.
        """
        pass

    def train(self, *args, **kwargs):
        """
        Exact matching doesn't require a training phase, so this method remains empty.
        """
        pass

    def cluster(self):
        df = self.df
        df['matchify_hash'] = df.apply(lambda row: hash(tuple(row[[x for x in df.columns if x not in self.ignored_columns]])), axis=1)
        df["predicted_group_id"] = df["matchify_hash"].rank(method="dense", ascending=True)
        self.is_clustered = True
        self.df = df
        return df

    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        """
        Predict the most likely match(es) for the input record using exact matching.
        """
        if not self.is_clustered:
            self.cluster()
        return_full_record = kwargs.get('return_full_record')
        only_matches = kwargs.get('only_matches')

        record_hash = hash(tuple(record[[x for x in record.index if x not in self.ignored_columns]]))
        # to save memory we add a new column to the original loaded dataset and delete it later
        results = self.df
        results["score"] = results["matchify_hash"].apply(lambda x: 1 if x == record_hash else 0)

        results = results[self.df["score"] == 1] if only_matches else results
        results = results[['id', 'score']] if not return_full_record else results

        return results.sort_values(by='score', ascending=False)
