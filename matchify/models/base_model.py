import abc
import nameparser
import phonenumbers
import usaddress
import pandas as pd
from dateutil import parser as date_parser
from tqdm import tqdm


class ERBaseModel(abc.ABC):
    def __init__(self, df: pd.DataFrame, ignored_columns=None, **kwargs):
        self.df = df
        self.ignored_columns = ignored_columns or []
        self.ignored_columns += ['matchify_hash', 'predicted_group_id', 'score']
        self.is_clustered = False
        self.model = None
        self.kwargs = kwargs

    @abc.abstractmethod
    def preprocess(self, df: pd.DataFrame, ignored_columns=None) -> pd.DataFrame:
        """
        Preprocess the input data. This method should be implemented by each specific model
        to handle data cleaning, normalization, and other preprocessing tasks.
        """
        pass

    def _normalize_name(self, name: str) -> str:
        return str(nameparser.HumanName(name).lower())

    def _normalize_phone(self, phone: str) -> str:
        try:
            parsed_phone = phonenumbers.parse(phone, None)
            return phonenumbers.format_number(parsed_phone, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            return ""

    def _normalize_address(self, address: str) -> str:
        try:
            parsed_address, _ = usaddress.tag(address)
            return " ".join(parsed_address.values()).lower()
        except Exception:
            return ""

    def _normalize_date(self, date: str) -> str:
        try:
            return date_parser.parse(date).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return ""

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model using the preprocessed data. This method should be implemented by
        each specific model to handle training, including blocking, clustering, or
        learning from labeled data.
        """
        pass

    @abc.abstractmethod
    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        """
        Predict the most likely match(es) for the input record. This method should be
        implemented by each specific model to handle the actual entity resolution task.
        """
        pass

    def fit_predict(self, record: pd.Series, *args, **kwargs) -> pd.DataFrame:
        """
        Train the model and predict matches for the input record. This method is a
        convenience method that calls train() and predict() in succession.
        """
        self.train(*args, **kwargs)
        return self.predict(record)

    def _normalize_score(self, score, min_score, max_score):
        if max_score == min_score:
            return 1
        else:
            return (score - min_score) / (max_score - min_score)

    def _get_actual_matches(self, id) -> pd.DataFrame:
        if 'group_id' not in self.df.columns:
            raise Exception('_get_actual_matches requires group_id column')
        df_matching_id = self.df[self.df['id'] == id][:1]
        if df_matching_id.empty:
            raise Exception(f"id {id} not found in original dataframe")
        group_id = list(df_matching_id['group_id'])[0]
        if not group_id:
            return []
        matches = self.df[self.df['group_id'] == group_id]
        return matches

    def mrr(self):
        """
        Calculate the mean reciprocal rank (MRR) of the model's predictions.

        The MRR is a measure of how well the model's predicted matches correspond to the actual
        matches in the ground truth data. It is calculated as the average reciprocal rank of
        the correct match for each lookup record.

        Returns:
            The MRR value as a float between 0 and 1, or 0 if there are no relevant results.
        """
        if 'group_id' not in self.df.columns:
            raise Exception('MRR requires group_id column')

        rr_list = []  # list of reciprocal ranks

        total_rows = len(self.df)
        progress_bar = tqdm(total=total_rows, ncols=100, desc="Calculating MRR", unit=" rows", unit_scale=True)

        for index, row in self.df.iterrows():
            row_id = row.id
            row = row[[x for x in row.index if x not in self.ignored_columns]]

            # Get predicted matches for the lookup row
            predicted_matches = self.predict(row, only_matches=True)

            # Skip to the next row if there are no predicted matches
            if predicted_matches.empty:
                progress_bar.update(1)
                continue

            predicted_match_ids = list(predicted_matches['id'])

            # Exclude the id of the lookup row
            predicted_match_ids = [x for x in predicted_match_ids if x != row_id]

            # Get actual matches from ground truth data for the lookup row
            actual_matches = self._get_actual_matches(row_id)
            actual_match_ids = list(actual_matches['id'])

            # Skip to the next row if there are no actual matches
            if not actual_match_ids:
                progress_bar.update(1)
                continue

            # Handle cases where either actual or predicted matches are empty
            if not predicted_match_ids:
                rr_list.append(0)
                progress_bar.update(1)
                continue

            # Calculate the reciprocal rank of the correct match
            for idx, pred_id in enumerate(predicted_match_ids):
                if pred_id in actual_match_ids and pred_id != 1:
                    rr_list.append(1 / (idx + 1))
                    break
            else:
                rr_list.append(0)

            progress_bar.update(1)

        progress_bar.close()

        # Calculate the mean reciprocal rank
        mrr = sum(rr_list) / len(rr_list) if rr_list else 0
        return mrr
