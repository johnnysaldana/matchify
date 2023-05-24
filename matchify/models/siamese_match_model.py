"""
Siamese entity-resolution model.

Fine-tunes a sentence-transformer with contrastive loss on positive
(same-group_id) and negative (different-group_id) pairs from the
training data, then ranks candidates by cosine similarity in the
fine-tuned encoder's embedding space.

Canonical twin-encoder Siamese (Mudgal et al., DeepER) on top of
sentence-transformers' SentenceTransformer + ContrastiveLoss.

Needs the [deep] extra: pip install matchify[deep].
"""

import os
import random
import tempfile

import pandas as pd

from matchify.models.base_model import ERBaseModel


class SiameseMatchModel(ERBaseModel):
    """
    Supervised embedding model. Fine-tunes a SentenceTransformer with
    contrastive loss on sampled positive/negative record pairs, then
    ranks candidates by cosine similarity in the learned space.

    Training pairs use group_id supervision from the train partition.
    Positives share a group_id, negatives do not.
    """
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        df: pd.DataFrame,
        field_config,
        blocking_config=None,
        ignored_columns=None,
        base_model: str = DEFAULT_MODEL,
        n_pairs: int = 4000,
        epochs: int = 1,
        batch_size: int = 32,
        margin: float = 0.5,
        random_state: int = 42,
        save_path: str = None,
    ):
        super().__init__(df, ignored_columns)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SiameseMatchModel requires the [deep] extra: "
                "pip install matchify[deep]"
            ) from e

        self.field_config = field_config
        self.blocking_config = blocking_config or {}
        self.blocking_key = (
            list(self.blocking_config.keys())[0] if self.blocking_config else None
        )
        self.base_model = base_model
        self.n_pairs = n_pairs
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = margin
        self.random_state = random_state
        self.save_path = save_path

        self.encoder = SentenceTransformer(base_model)
        self.preprocessed_data = self.preprocess(df)
        self.embeddings = None

    def preprocess(self, df) -> pd.DataFrame:
        # Add derived fields needed for blocking.
        df = df.copy()
        for field, cfg in self.blocking_config.items():
            if cfg.get('method') == 'prefix':
                prefix_len = cfg['threshold']
                prefix_field = f"{field}_prefix_{prefix_len}"
                self.blocking_config[field]['field'] = prefix_field
                df[prefix_field] = df[field].fillna('').astype(str).str[:prefix_len]
        return df

    def _record_text(self, record) -> str:
        # Build a single text representation from configured fields.
        parts = []
        for field in self.field_config:
            v = record.get(field)
            if pd.isna(v):
                continue
            parts.append(f"{field}: {v}")
        return " | ".join(parts)

    def _sample_pairs(self):
        # Sample positive/negative supervision pairs from train rows.
        if 'group_id' not in self.df.columns:
            raise Exception('SiameseMatchModel requires group_id supervision')
        rng = random.Random(self.random_state)
        # restrict pair sampling to the train partition. when test_size=0
        # this is the full df
        train_df = self.df.loc[self._train_idx]

        groups = df.dropna(subset=['group_id']).groupby('group_id').indices
        candidate_groups = [list(idxs) for idxs in groups.values() if len(idxs) >= 2]
        all_indices = df.index.to_list()
        group_lookup = df['group_id'].to_dict()

        n_each = self.n_pairs // 2
        positives, negatives = [], []
        attempts = 0
        while len(positives) < n_each and attempts < n_each * 10 and candidate_groups:
            attempts += 1
            grp = rng.choice(candidate_groups)
            i, j = rng.sample(grp, 2)
            positives.append((i, j))
        attempts = 0
        while len(negatives) < n_each and attempts < n_each * 20:
            attempts += 1
            i, j = rng.sample(all_indices, 2)
            gi, gj = group_lookup.get(i), group_lookup.get(j)
            if pd.isna(gi) or pd.isna(gj) or gi != gj:
                negatives.append((i, j))
        return positives, negatives

    def train(self):
        # Fine-tune the encoder on sampled supervision pairs.
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        positives, negatives = self._sample_pairs()
        examples = []
        for i, j in positives:
            examples.append(InputExample(
                texts=[self._record_text(self.df.loc[i]), self._record_text(self.df.loc[j])],
                label=1.0,
            ))
        for i, j in negatives:
            examples.append(InputExample(
                texts=[self._record_text(self.df.loc[i]), self._record_text(self.df.loc[j])],
                label=0.0,
            ))
        random.Random(self.random_state).shuffle(examples)

        loader = DataLoader(examples, shuffle=True, batch_size=self.batch_size)
        loss = losses.ContrastiveLoss(model=self.encoder, margin=self.margin)
        warmup = max(1, int(len(loader) * self.epochs * 0.1))

        save_dir = self.save_path or tempfile.mkdtemp(prefix='siamese_')
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.fit(
            train_objectives=[(loader, loss)],
            epochs=self.epochs,
            warmup_steps=warmup,
            output_path=save_dir,
            show_progress_bar=False,
        )
        self.save_path = save_dir

        # cache embeddings under the fine-tuned encoder
        texts = [self._record_text(row) for _, row in self.df.iterrows()]
        self.embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _apply_blocking(self, record):
        # Restrict candidate rows according to configured blocking strategy.
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
        # Score blocked candidates with cosine similarity in embedding space.
        if self.embeddings is None:
            raise Exception('Call train() before predict()')
        only_matches = kwargs.get('only_matches')
        return_full_record = kwargs.get('return_full_record')

        q_emb = self.encoder.encode(
            [self._record_text(record)],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        candidate_indices = self._apply_blocking(record)
        positions = [self.df.index.get_loc(i) for i in candidate_indices]
        scores = self.embeddings[positions] @ q_emb

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
