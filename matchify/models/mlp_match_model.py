import random

import jellyfish
import numpy as np
import pandas as pd
import textdistance
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier

from matchify.models.base_model import ERBaseModel


class MLPMatchModel(ERBaseModel):
    """
    Supervised model. Turns each candidate pair into a fixed feature
    vector of per-field similarity scores and feeds it to an
    MLPClassifier. Match probability is the ranking score.

    Training pairs come from the same group_id supervision as MRR.
    Positives share a group_id, negatives don't. 50/50 by default.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        field_config: dict[str, dict[str, str]],
        blocking_config: dict[str, dict[str, str]],
        ignored_columns,
        n_pairs: int = 4000,
        hidden_layer_sizes=(64, 32),
        max_iter: int = 200,
        random_state: int = 42,
    ):
        super().__init__(df, ignored_columns)
        self.field_config = field_config
        self.blocking_config = blocking_config
        self.blocking_key = list(blocking_config.keys())[0]
        self.n_pairs = n_pairs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state

        self.preprocessed_data = self.preprocess(df)
        self.vectorizers = {}
        # tfidf vector and jaccard token cache. populated in train() so
        # pair feature extraction reuses them instead of recomputing.
        self._tfidf_cache = {}
        self._token_cache = {}
        self.classifier = None

    def preprocess(self, df) -> pd.DataFrame:
        preprocessed_data = df.copy()

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

        for field, config in self.blocking_config.items():
            if config['method'] == 'prefix':
                prefix_len = config['threshold']
                prefix_field = f"{field}_prefix_{prefix_len}"
                self.blocking_config[field]['field'] = prefix_field
                # bind prefix_len explicitly; otherwise the lambda captures the loop variable
                preprocessed_data[prefix_field] = preprocessed_data[field].apply(
                    lambda x, p=prefix_len: str(x)[:p]
                )
        return preprocessed_data

    def _fit_vectorizers(self):
        for field in self.field_config:
            methods = self._feature_methods(field)
            col = self.df[field].fillna('').astype(str)
            if 'tfidf_cosine' in methods:
                cleaned = col[col.str.len() > 0].values
                if len(cleaned) > 0:
                    vec = TfidfVectorizer()
                    vec.fit(cleaned)
                    self.vectorizers[field] = vec
                    # vectorize every record once
                    matrix = vec.transform(col.values)
                    for idx, vector_row in zip(self.df.index, matrix):
                        self._tfidf_cache[(idx, field)] = vector_row
            if 'jaccard' in methods:
                # tokenize every record once
                for idx, value in zip(self.df.index, col.values):
                    self._token_cache[(idx, field)] = set(
                        preprocess_string(value, filters=[strip_punctuation])
                    )

    def _feature_methods(self, field):
        # 4 similarity features per field by default.
        # override via field_config[field]['features'].
        return self.field_config[field].get(
            'features', ['jaro_winkler', 'levenshtein', 'tfidf_cosine', 'jaccard']
        )

    def _pair_features(self, row_a: pd.Series, row_b: pd.Series) -> np.ndarray:
        feats = []
        idx_a = row_a.name if isinstance(row_a, pd.Series) else None
        idx_b = row_b.name if isinstance(row_b, pd.Series) else None
        for field in self.field_config:
            v1, v2 = row_a.get(field), row_b.get(field)
            s1 = "" if pd.isna(v1) else str(v1)
            s2 = "" if pd.isna(v2) else str(v2)
            for method in self._feature_methods(field):
                feats.append(self._compare(s1, s2, method, field, idx_a, idx_b))
        return np.asarray(feats, dtype=float)

    def _compare(self, s1: str, s2: str, method: str, field: str, idx_a=None, idx_b=None) -> float:
        if not s1 or not s2:
            return 0.0
        if method == 'jaro_winkler':
            return jellyfish.jaro_winkler_similarity(s1, s2)
        if method == 'levenshtein':
            return textdistance.levenshtein.normalized_similarity(s1, s2)
        if method == 'tfidf_cosine':
            vec = self.vectorizers.get(field)
            if vec is None:
                return 0.0
            t1 = self._tfidf_cache.get((idx_a, field))
            if t1 is None:
                t1 = vec.transform([s1])
            t2 = self._tfidf_cache.get((idx_b, field))
            if t2 is None:
                t2 = vec.transform([s2])
            return float(cosine_similarity(t1, t2)[0][0])
        if method == 'jaccard':
            tok_a = self._token_cache.get((idx_a, field))
            if tok_a is None:
                tok_a = set(preprocess_string(s1, filters=[strip_punctuation]))
            tok_b = self._token_cache.get((idx_b, field))
            if tok_b is None:
                tok_b = set(preprocess_string(s2, filters=[strip_punctuation]))
            union = tok_a | tok_b
            return len(tok_a & tok_b) / len(union) if union else 0.0
        raise ValueError(f"Unsupported comparison method: {method}")

    def _sample_training_pairs(self):
        """Pull positive and negative pairs from group_id supervision."""
        if 'group_id' not in self.df.columns:
            raise Exception('MLPMatchModel requires a group_id column for training')

        rng = random.Random(self.random_state)
        df = self.df

        # positives: two distinct rows in the same group
        groups = df.dropna(subset=['group_id']).groupby('group_id').indices
        candidate_groups = [idxs for idxs in groups.values() if len(idxs) >= 2]

        n_each = self.n_pairs // 2
        positives = []
        attempts = 0
        while len(positives) < n_each and attempts < n_each * 10:
            attempts += 1
            if not candidate_groups:
                break
            grp = rng.choice(candidate_groups)
            i, j = rng.sample(list(grp), 2)
            positives.append((i, j))

        # negatives: two rows from different groups
        all_indices = df.index.to_list()
        group_lookup = df['group_id'].to_dict()
        negatives = []
        attempts = 0
        while len(negatives) < n_each and attempts < n_each * 20:
            attempts += 1
            i, j = rng.sample(all_indices, 2)
            gi, gj = group_lookup.get(i), group_lookup.get(j)
            if pd.isna(gi) or pd.isna(gj) or gi != gj:
                negatives.append((i, j))

        return positives, negatives

    def train(self):
        self._fit_vectorizers()
        positives, negatives = self._sample_training_pairs()
        if not positives or not negatives:
            raise Exception('Could not sample any training pairs - check group_id supervision')

        X = []
        y = []
        for i, j in positives:
            X.append(self._pair_features(self.preprocessed_data.loc[i], self.preprocessed_data.loc[j]))
            y.append(1)
        for i, j in negatives:
            X.append(self._pair_features(self.preprocessed_data.loc[i], self.preprocessed_data.loc[j]))
            y.append(0)

        X = np.vstack(X)
        y = np.asarray(y)

        self.classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.classifier.fit(X, y)

    def _apply_blocking(self, new_record: pd.Series) -> pd.Index:
        preprocessed = self.preprocess(new_record.to_frame().T).iloc[0]
        config = self.blocking_config[self.blocking_key]
        method = config['method']
        threshold = config['threshold']
        blocking_field = config['field']
        key = preprocessed[blocking_field]

        if method == 'prefix' or method == 'block':
            return self.preprocessed_data[
                self.preprocessed_data[blocking_field] == key
            ].index
        if method == 'sorted_neighborhood':
            sorted_data = self.preprocessed_data.sort_values(by=blocking_field)
            pos = sorted_data[blocking_field].searchsorted(key)
            return sorted_data.iloc[max(0, pos - threshold): pos + threshold].index
        if method == 'full':
            return self.preprocessed_data.index
        raise ValueError(f"Unsupported blocking method: {method}")

    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        if self.classifier is None:
            raise Exception('Call train() before predict()')

        preprocessed_record = self.preprocess(record.to_frame().T).iloc[0]
        candidate_indices = self._apply_blocking(record)
        if len(candidate_indices) == 0:
            empty = self.df.iloc[0:0].copy()
            empty['score'] = []
            return empty

        # Score all candidates in one batched predict_proba call.
        feats = np.vstack([
            self._pair_features(preprocessed_record, self.preprocessed_data.loc[idx])
            for idx in candidate_indices
        ])
        probs = self.classifier.predict_proba(feats)[:, 1]

        scores_df = pd.DataFrame({'Index': list(candidate_indices), 'score': probs})
        sorted_scores = scores_df.sort_values(by='score', ascending=False).drop_duplicates().reset_index(drop=True)

        best_matches = self.df.loc[sorted_scores['Index']].reset_index(drop=True)
        sorted_scores = sorted_scores.drop(columns='Index').reset_index(drop=True)
        return pd.concat([best_matches, sorted_scores], axis=1)
