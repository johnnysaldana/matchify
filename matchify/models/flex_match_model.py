import pandas as pd
import textdistance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation
import numpy as np
from matchify.models.base_model import ERBaseModel
from typing import Dict


class FlexMatchModel(ERBaseModel):
    def __init__(self, df: pd.DataFrame, field_config: Dict[str, Dict[str, str]], blocking_config: Dict[str, Dict[str, str]], ignored_columns):
        super().__init__(df, ignored_columns)
        self.field_config = field_config
        self.blocking_config = blocking_config
        self.blocking_key = list(blocking_config.keys())[0]
        self.preprocessed_data = self.preprocess(df)

        self.models = {}
        self.max_score = len(list(field_config.keys()))

    def preprocess(self, df) -> pd.DataFrame:
        # Preprocess data based on field types
        preprocessed_data = df

        for field, config in self.field_config.items():
            field_type = config["type"]
            if field_type == "name":
                preprocessed_data[field] = preprocessed_data[field].apply(self._normalize_name)
            elif field_type == "phone":
                preprocessed_data[field] = preprocessed_data[field].apply(self._normalize_phone)
            elif field_type == "address":
                preprocessed_data[field] = preprocessed_data[field].apply(self._normalize_address)
            elif field_type == "date":
                preprocessed_data[field] = preprocessed_data[field].apply(self._normalize_date)
            elif field_type == "description":
                preprocessed_data[field] = preprocessed_data[field].apply(self._normalize_description)

        for field, config in self.blocking_config.items():
            # add necessary fields for blocking purposes
            if config['method'] == 'prefix':
                prefix_len = config['threshold']
                prefix_field = f"{field}_prefix_{prefix_len}"
                self.blocking_config[field]['field'] = prefix_field
                preprocessed_data[prefix_field] = preprocessed_data[field].apply(lambda x: x[:prefix_len])
        return preprocessed_data

    def train(self):
        for field, config in self.field_config.items():
            if config['comparison_method'] == "tfidf_cosine":
                cleaned_column = self.df[field][self.df[field].astype(bool)].values.astype('U')
                vectorizer = TfidfVectorizer()
                vectorizer.fit(cleaned_column)
                self.models[field] = vectorizer

    def tfidf_cosine_similarity(self, s1, s2, vectorizer):
        # Transform the descriptions using the fitted TfidfVectorizer
        str1_tfidf = vectorizer.transform([s1])
        str2_tfidf = vectorizer.transform([s2])

        # Calculate the cosine similarity between the two transformed descriptions
        return cosine_similarity(str1_tfidf, str2_tfidf)[0][0]

    # Comparison methods
    def _compare_strings(self, s1: str, s2: str, method: str, column: str) -> float:
        s1 = str(s1) or ""
        s2 = str(s2) or ""
        if method == 'jaro_winkler':
            return textdistance.jaro_winkler(s1, s2)
        elif method == 'levenshtein':
            return textdistance.levenshtein.normalized_similarity(s1, s2)
        elif method == 'tfidf_cosine':
            return self.tfidf_cosine_similarity(s1, s2, self.models[column])
        elif method == 'jaccard_similarity':
            tokenized_s1 = set(preprocess_string(s1, filters=[strip_punctuation]))
            tokenized_s2 = set(preprocess_string(s2, filters=[strip_punctuation]))
            intersection = len(tokenized_s1.intersection(tokenized_s2))
            union = len(tokenized_s1.union(tokenized_s2))
            return float(intersection) / float

    # Blocking methods
    def _apply_blocking(self, new_record: pd.Series) -> pd.MultiIndex:
        preprocessed_new_record = self.preprocess(new_record.to_frame().T).iloc[0]
        config = self.blocking_config[self.blocking_key]
        method = config['method']
        threshold = config['threshold']
        blocking_field = config['field']
        new_record_blocking_value = preprocessed_new_record[blocking_field]

        if method == 'sorted_neighborhood':
            # Sort the dataset based on the blocking field
            sorted_data = self.preprocessed_data.sort_values(by=blocking_field)
            # Find the position where the new record should be inserted
            insert_position = sorted_data[blocking_field].searchsorted(new_record_blocking_value)
            # Define the range of records around the new record's value
            range_start = max(0, insert_position - threshold)
            range_end = min(len(sorted_data), insert_position + threshold)
            # Get the indices of the candidates
            candidate_indices = sorted_data.iloc[range_start:range_end].index
        elif method == 'prefix':
            # Filter the dataset based on the new record's prefix
            candidates = self.preprocessed_data[self.preprocessed_data[blocking_field] == new_record_blocking_value]
            # Get the indices of the candidates
            candidate_indices = candidates.index
        elif method == 'block':
            # Filter the dataset based on the new record's value
            candidates = self.preprocessed_data[self.preprocessed_data[blocking_field] == new_record_blocking_value]
            # Get the indices of the candidates
            candidate_indices = candidates.index
        elif method == 'full':
            candidates = self.preprocessed_data
            candidate_indices = self.preprocessed_data.index
        # Add more blocking methods here
        else:
            raise ValueError(f"Unsupported blocking method: {method}")

        return candidate_indices

    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        preprocessed_record = self.preprocess(record.to_frame().T).iloc[0]
        # Apply the blocking method specified by the user
        candidate_indices = self._apply_blocking(record)
        # Calculate similarity scores for each pair
        scores = []
        for candidate_id in candidate_indices:
            record_b = self.preprocessed_data.loc[candidate_id]

            similarity = 0
            for field, config in self.field_config.items():
                comparison_method = config['comparison_method']
                field_similarity = self._compare_strings(preprocessed_record[field], record_b[field], comparison_method, field)
                similarity += field_similarity
            similarity = self._normalize_score(similarity, min_score=0, max_score=self.max_score)
            scores.append((candidate_id, similarity))

        scores_df = pd.DataFrame(scores, columns=['Index', 'score'])
        sorted_scores_df = scores_df.sort_values(by='score', ascending=False).drop_duplicates().reset_index(drop=True)

        # Add original records to the result DataFrame
        best_matches = self.df.loc[sorted_scores_df['Index']].reset_index(drop=True)
        sorted_scores_df = sorted_scores_df.drop(columns='Index').reset_index(drop=True)
        result_df = pd.concat([best_matches, sorted_scores_df], axis=1)

        return result_df
