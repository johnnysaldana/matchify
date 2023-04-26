"""
BERT-based entity-resolution model using sentence-transformers.

Each record's configured fields are concatenated into a single text and
encoded once (cached on the model), then candidate matches are ranked by
cosine similarity in embedding space. Optionally a blocking step prunes
the candidate set first.

Requires the [deep] extra: `pip install matchify[deep]`.
"""

import numpy as np
import pandas as pd

from matchify.models.base_model import ERBaseModel


class BertMatchModel(ERBaseModel):
    """
    Unsupervised embedding model. Encodes each record into a sentence
    vector with SentenceTransformers and ranks candidates by cosine
    similarity against the query embedding.

    Blocking is optional and limits candidate scoring before the
    embedding dot-product step.
    """
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        df: pd.DataFrame,
        field_config,
        blocking_config=None,
        ignored_columns=None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
    ):
        super().__init__(df, ignored_columns)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "BertMatchModel requires the [deep] extra: "
                "pip install matchify[deep]"
            ) from e

        self.field_config = field_config
        self.blocking_config = blocking_config or {}
        self.blocking_key = (
            list(self.blocking_config.keys())[0] if self.blocking_config else None
        )
        self.model_name = model_name
        self.batch_size = batch_size

        self.encoder = SentenceTransformer(model_name)
        self.preprocessed_data = self.preprocess(df)
        self.embeddings = None

    def preprocess(self, df) -> pd.DataFrame:
        df = df.copy()
        for field, cfg in self.blocking_config.items():
            if cfg.get('method') == 'prefix':
                prefix_len = cfg['threshold']
                prefix_field = f"{field}_prefix_{prefix_len}"
                self.blocking_config[field]['field'] = prefix_field
                df[prefix_field] = df[field].fillna('').astype(str).str[:prefix_len]
        return df

    def _record_text(self, record) -> str:
        parts = []
        for field in self.field_config:
            v = record.get(field)
            if pd.isna(v):
                continue
            parts.append(f"{field}: {v}")
        return " | ".join(parts)

    def train(self):
        """Encode every record in the dataset once and cache the embedding matrix."""
        texts = [self._record_text(row) for _, row in self.df.iterrows()]
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _apply_blocking(self, record):
        if not self.blocking_config or self.blocking_key is None:
            return self.preprocessed_data.index
        cfg = self.blocking_config[self.blocking_key]
        method = cfg['method']
        blocking_field = cfg['field']
        key = self.preprocess(record.to_frame().T).iloc[0][blocking_field]
        if method in ('prefix', 'block'):
            return self.preprocessed_data[
                self.preprocessed_data[blocking_field] == key
            ].index
        if method == 'full':
            return self.preprocessed_data.index
        raise ValueError(f"Unsupported blocking method: {method}")

    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        if self.embeddings is None:
            raise Exception('Call train() before predict()')

        only_matches = kwargs.get('only_matches')
        return_full_record = kwargs.get('return_full_record')

        text = self._record_text(record)
        q_emb = self.encoder.encode(
            [text], convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        candidate_indices = self._apply_blocking(record)
        positions = [self.df.index.get_loc(i) for i in candidate_indices]
        cand_embs = self.embeddings[positions]

        # dot product is cosine similarity
        scores = cand_embs @ q_emb

        results = pd.DataFrame({
            'index': candidate_indices,
            'score': scores,
        }).sort_values(by='score', ascending=False).reset_index(drop=True)

        records = self.df.loc[results['index']].reset_index(drop=True)
        out = pd.concat([records, results[['score']]], axis=1)

        if only_matches:
            out = out[out['score'] >= 0.5]
        if not return_full_record:
            out = out[['id', 'score']]
        return out
