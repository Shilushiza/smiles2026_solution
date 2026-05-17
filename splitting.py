from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    idx = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    splits = []
    for train_rel, val_rel in skf.split(idx_trainval, y[idx_trainval]):
        idx_tr = idx_trainval[train_rel]
        idx_va = idx_trainval[val_rel]
        splits.append((idx_tr, idx_va, idx_test))

    return splits
