import abc
import random

import nameparser
import pandas as pd
import phonenumbers
import usaddress
from dateutil import parser as date_parser
from tqdm import tqdm


class ERBaseModel(abc.ABC):
    def __init__(
        self,
        df: pd.DataFrame,
        ignored_columns=None,
        test_size: float = 0.0,
        random_state: int = 0,
        **kwargs,
    ):
        self.df = df
        self.ignored_columns = list(ignored_columns or [])
        self.ignored_columns += ['matchify_hash', 'predicted_group_id', 'score']
        self.is_clustered = False
        self.model = None
        self.kwargs = kwargs
        self.test_size = test_size
        self.random_state = random_state
        self._train_idx, self._test_idx = self._compute_split()

    def _compute_split(self):
        # split groups (not rows) so supervised models never see test pairs.
        # singletons and NaN group_id rows go to train. they have no positive
        # match to find. test_size=0 returns an empty test partition.
        if not self.test_size or self.test_size <= 0 or 'group_id' not in self.df.columns:
            return self.df.index, pd.Index([])

        group_sizes = self.df.dropna(subset=['group_id']).groupby('group_id').size()
        multi_groups = group_sizes[group_sizes >= 2].index.tolist()
        if not multi_groups:
            return self.df.index, pd.Index([])

        rng = random.Random(self.random_state)
        rng.shuffle(multi_groups)
        n_test = max(1, int(round(len(multi_groups) * self.test_size)))
        test_groups = set(multi_groups[:n_test])

        test_mask = self.df['group_id'].isin(test_groups)
        return self.df.index[~test_mask], self.df.index[test_mask]

    @property
    def _eval_idx(self):
        # test rows when a split is configured, else the full df.
        return self._test_idx if len(self._test_idx) > 0 else self.df.index

    @abc.abstractmethod
    def preprocess(self, df: pd.DataFrame, ignored_columns=None) -> pd.DataFrame:
        pass

    def _normalize_name(self, name: str) -> str:
        # .lower() goes on the string, not on the HumanName instance
        return str(nameparser.HumanName(name)).lower()

    def _normalize_phone(self, phone: str) -> str:
        # phonenumbers.parse needs a region for non-international numbers.
        # default to US so "(555) 123-4567" works. fall back to digits.
        try:
            parsed_phone = phonenumbers.parse(phone, "US")
            return phonenumbers.format_number(parsed_phone, phonenumbers.PhoneNumberFormat.E164)
        except phonenumbers.NumberParseException:
            digits = "".join(ch for ch in str(phone) if ch.isdigit())
            return digits

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
        pass

    @abc.abstractmethod
    def predict(self, record: pd.Series, **kwargs) -> pd.DataFrame:
        pass

    def fit_predict(self, record: pd.Series, *args, **kwargs) -> pd.DataFrame:
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
        """Mean reciprocal rank over eval rows. Reciprocal rank of the first
        true positive in each predicted candidate list, averaged."""
        if 'group_id' not in self.df.columns:
            raise Exception('MRR requires group_id column')

        rr_list = []
        eval_idx = self._eval_idx
        progress_bar = tqdm(total=len(eval_idx), ncols=100, desc="Calculating MRR", unit=" rows", unit_scale=True)

        for index in eval_idx:
            row = self.df.loc[index]
            row_id = row.id
            row = row[[x for x in row.index if x not in self.ignored_columns]]

            predicted_matches = self.predict(row, only_matches=True)
            if predicted_matches.empty:
                progress_bar.update(1)
                continue

            predicted_match_ids = [x for x in predicted_matches['id'] if x != row_id]

            actual_match_ids = list(self._get_actual_matches(row_id)['id'])
            if not actual_match_ids:
                progress_bar.update(1)
                continue

            if not predicted_match_ids:
                rr_list.append(0)
                progress_bar.update(1)
                continue

            for idx, pred_id in enumerate(predicted_match_ids):
                if pred_id in actual_match_ids:
                    rr_list.append(1 / (idx + 1))
                    break
            else:
                rr_list.append(0)

            progress_bar.update(1)

        progress_bar.close()
        return sum(rr_list) / len(rr_list) if rr_list else 0

    def _score_all_pairs(self):
        # walk eval rows once, collect (score, is_match) per candidate pair.
        # shared between confusion_matrix and threshold_sweep. cache the
        # result so repeated callers (e.g., sweep + confusion at one
        # threshold) don't re-run predict over every eval row.
        if getattr(self, '_pair_score_cache', None) is not None:
            return self._pair_score_cache
        if 'group_id' not in self.df.columns:
            raise Exception('scoring requires group_id column')

        group_lookup = self.df.set_index('id')['group_id'].to_dict()
        scores = []
        labels = []

        eval_idx = self._eval_idx
        progress_bar = tqdm(total=len(eval_idx), ncols=100, desc="Scoring pairs", unit=" rows")
        for index in eval_idx:
            row = self.df.loc[index]
            row_id = row.id
            row_group = group_lookup.get(row_id)
            row_features = row[[x for x in row.index if x not in self.ignored_columns]]
            preds = self.predict(row_features, only_matches=False, return_full_record=True)
            for _, pred in preds.iterrows():
                pred_id = pred.get('id')
                if pred_id == row_id:
                    continue
                pred_group = group_lookup.get(pred_id)
                actual_match = (
                    row_group is not None
                    and pred_group is not None
                    and not pd.isna(row_group)
                    and not pd.isna(pred_group)
                    and row_group == pred_group
                )
                scores.append(float(pred.get('score') or 0.0))
                labels.append(bool(actual_match))
            progress_bar.update(1)
        progress_bar.close()
        self._pair_score_cache = (scores, labels)
        return scores, labels

    @staticmethod
    def _stats_at_threshold(scores, labels, threshold):
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels):
            predicted_match = s >= threshold
            if predicted_match and y:
                tp += 1
            elif predicted_match and not y:
                fp += 1
            elif not predicted_match and y:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            'threshold': threshold,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }

    def confusion_matrix(self, threshold: float = 0.5):
        """TP/FP/TN/FN at one threshold, over candidate pairs from predict."""
        scores, labels = self._score_all_pairs()
        return self._stats_at_threshold(scores, labels, threshold)

    def threshold_sweep(self, thresholds=None) -> pd.DataFrame:
        """confusion_matrix at many thresholds, predict only runs once per row.

        With thresholds=None, sweep at the distinct observed scores
        (capped at 200 quantile-thinned points so plots stay readable
        when scores are dense). A fixed grid loses resolution when scores
        cluster, which produces degenerate PR curves for models whose
        outputs are bimodal or live in a narrow band.
        """
        import numpy as np
        scores, labels = self._score_all_pairs()
        if thresholds is None:
            arr = np.asarray(scores, dtype=float)
            uniq = np.unique(arr) if arr.size else np.array([0.0])
            if len(uniq) > 200:
                uniq = np.unique(np.quantile(uniq, np.linspace(0.0, 1.0, 200)))
            sentinel = float(uniq[-1]) + 1e-9
            thresholds = np.unique(np.concatenate(([0.0], uniq, [sentinel])))
        rows = [self._stats_at_threshold(scores, labels, float(t)) for t in thresholds]
        return pd.DataFrame(rows)
