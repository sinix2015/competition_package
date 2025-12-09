from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from .features import FeatureGenerator


def build_training_arrays(
    dataset: pd.DataFrame,
    state_dim: int,
    *,
    feature_generator_cls: Type[FeatureGenerator] = FeatureGenerator,
    feature_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_list = []
    target_list = []
    seq_indices = []
    step_indices = []

    grouped = dataset.sort_values(["seq_ix", "step_in_seq"]).groupby("seq_ix", sort=False)
    state_columns = dataset.columns[3:]

    feature_kwargs = feature_kwargs or {}

    for seq_ix, group in grouped:
        generator = feature_generator_cls(state_dim=state_dim, **feature_kwargs)

        states = group[state_columns].to_numpy(dtype=np.float32)
        needs = group["need_prediction"].to_numpy(dtype=bool)
        steps = group["step_in_seq"].to_numpy(dtype=np.int32)

        prev_features = None
        prev_need_prediction = False

        for idx in range(len(group)):
            state = states[idx]
            step_value = int(steps[idx])

            features = generator.update(state, step=step_value, seq_ix=int(seq_ix))

            if prev_need_prediction and prev_features is not None:
                target_list.append(state)
                feature_list.append(prev_features)
                seq_indices.append(int(seq_ix))
                step_indices.append(step_value)

            prev_features = features
            prev_need_prediction = bool(needs[idx])

    features_array = np.stack(feature_list, axis=0)
    targets_array = np.stack(target_list, axis=0)
    seq_array = np.array(seq_indices, dtype=np.int32)
    step_array = np.array(step_indices, dtype=np.int32)
    return features_array, targets_array, seq_array, step_array


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    try:
        table = pq.read_table(dataset_path, use_threads=False)
    except OSError:
        try:
            parquet_file = pq.ParquetFile(dataset_path)
            table = parquet_file.read()
        except OSError:
            df = pd.read_parquet(dataset_path, engine="fastparquet")
            return df.reset_index(drop=True)
    df = table.to_pandas()
    return df.reset_index(drop=True)
