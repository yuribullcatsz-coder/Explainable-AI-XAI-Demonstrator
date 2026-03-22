"""Fairness-style metrics without requiring AIF360 (optional AIF360 later)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BiasReport:
    """Simple group fairness statistics for binary classification."""

    positive_rate_group_a: float
    positive_rate_group_b: float
    statistical_parity_difference: float
    disparate_impact_ratio: float | None
    n_group_a: int
    n_group_b: int
    message: str | None = None


def _binary_positive(pred: np.ndarray, positive_label: str | None) -> np.ndarray:
    """Return boolean array: predicted positive class."""
    if pred.dtype == object or pred.dtype.kind in ("U", "S", "O"):
        u = np.unique(pred)
        if positive_label is not None:
            pos = positive_label
        else:
            pos = sorted([str(x) for x in u])[-1]
        return pred.astype(str) == str(pos)
    # numeric predictions treated as scores / classes
    return pred.astype(float) >= 0.5


def compute_group_metrics(
    y_pred: np.ndarray,
    sensitive: pd.Series,
    positive_label: str | None = None,
) -> BiasReport:
    """
    Statistical parity difference and disparate impact for two groups.

    sensitive must be binary (exactly two unique values after dropna alignment).
    """
    s = sensitive.astype(str)
    levels = sorted(s.dropna().unique().tolist())
    if len(levels) != 2:
        return BiasReport(
            positive_rate_group_a=float("nan"),
            positive_rate_group_b=float("nan"),
            statistical_parity_difference=float("nan"),
            disparate_impact_ratio=None,
            n_group_a=0,
            n_group_b=0,
            message="Sensitive attribute must have exactly two groups (after dropping NaN).",
        )

    mask = s.notna()
    y_p = np.asarray(y_pred)[mask.values]
    s_clean = s[mask]

    g0, g1 = levels[0], levels[1]
    m0 = s_clean == g0
    m1 = s_clean == g1

    pos = _binary_positive(y_p, positive_label)
    r0 = float(np.mean(pos[m0.values])) if m0.any() else float("nan")
    r1 = float(np.mean(pos[m1.values])) if m1.any() else float("nan")

    spd = r1 - r0 if not (np.isnan(r0) or np.isnan(r1)) else float("nan")
    if r0 == 0:
        di = None if r1 > 0 else 1.0
    else:
        di = r1 / r0

    return BiasReport(
        positive_rate_group_a=r0,
        positive_rate_group_b=r1,
        statistical_parity_difference=spd,
        disparate_impact_ratio=di,
        n_group_a=int(m0.sum()),
        n_group_b=int(m1.sum()),
        message=None,
    )


def predictions_for_frame(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))
