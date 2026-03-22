"""SHAP, partial dependence, and LIME helpers."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder

from xai_demo.pipeline import model_matrix_for_explain


def _lime_numeric_copy(
    X_train: pd.DataFrame, X_row: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode non-numeric columns so LIME can perturb numeric arrays."""
    X_tr = X_train.copy()
    X_rw = X_row.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in X_tr.columns:
        if X_tr[col].dtype == object or pd.api.types.is_categorical_dtype(X_tr[col]):
            le = LabelEncoder()
            X_tr[col] = le.fit_transform(X_tr[col].astype(str))
            X_rw[col] = le.transform(X_rw[col].astype(str))
            encoders[col] = le
        else:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
            X_rw[col] = pd.to_numeric(X_rw[col], errors="coerce")
    return X_tr, X_rw, encoders


def _decode_lime_batch(
    arr: np.ndarray, columns: list[str], encoders: dict[str, LabelEncoder]
) -> pd.DataFrame:
    df = pd.DataFrame(arr, columns=columns)
    for col in columns:
        if col in encoders:
            le = encoders[col]
            idx = np.clip(np.round(df[col].astype(float)).astype(int), 0, len(le.classes_) - 1)
            df[col] = le.inverse_transform(idx)
        else:
            df[col] = df[col].astype(float)
    return df


def shap_summary_plot(
    model: Any,
    X_sample: pd.DataFrame,
    max_samples: int = 400,
) -> tuple[bytes, str | None]:
    """
    Build SHAP summary plot for the tree model on preprocessed features.

    Returns PNG bytes and an error message (if any).
    """
    try:
        Xm, names = model_matrix_for_explain(model, X_sample)
        n = min(max_samples, Xm.shape[0])
        if n < 1:
            return b"", "No rows to explain."
        rng = np.random.default_rng(42)
        idx = rng.choice(Xm.shape[0], size=n, replace=False)
        Xm_s = Xm[idx]

        est = model.named_steps["model"]
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(Xm_s)
        if isinstance(shap_values, list):
            # Multiclass: average |SHAP| across classes for one summary view
            shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            shap_values = np.asarray(shap_values)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            Xm_s,
            feature_names=names,
            show=False,
            max_display=min(20, len(names)),
        )
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        return buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001 — surface to UI
        plt.close("all")
        return b"", str(exc)


def partial_dependence_figure(
    model: Any,
    X_train: pd.DataFrame,
    feature_name: str,
    grid_resolution: int = 20,
) -> tuple[bytes, str | None]:
    """
    Partial dependence for one original (pre-pipeline) column.

    feature_name must be a column in X_train.
    """
    try:
        if feature_name not in X_train.columns:
            return b"", f"Feature {feature_name!r} not in training columns."

        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(
            model,
            X_train,
            features=[feature_name],
            grid_resolution=grid_resolution,
            ax=ax,
        )
        ax.set_title(f"Partial dependence: {feature_name}")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return b"", str(exc)


def lime_explanation_figure(
    model: Any,
    X_train: pd.DataFrame,
    X_row: pd.DataFrame,
    task: str,
    num_features: int = 10,
) -> tuple[bytes, str | None]:
    """LIME tabular explanation for a single row (original feature columns)."""
    try:
        if X_row.shape[0] != 1:
            return b"", "LIME expects exactly one instance row."

        feature_names = list(X_train.columns)
        X_tr_num, X_rw_num, encoders = _lime_numeric_copy(X_train, X_row)
        train_np = np.asarray(X_tr_num, dtype=float)
        row_np = np.asarray(X_rw_num, dtype=float).ravel()
        if np.isnan(train_np).any() or np.isnan(row_np).any():
            return (
                b"",
                "LIME cannot run with NaN values in features. Impute or drop missing rows.",
            )

        mode = "classification" if task == "classification" else "regression"
        explainer = LimeTabularExplainer(
            train_np,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
        )

        def predict_proba_np(x: np.ndarray) -> np.ndarray:
            df = _decode_lime_batch(x, feature_names, encoders)
            return np.asarray(model.predict_proba(df))

        def predict_np(x: np.ndarray) -> np.ndarray:
            df = _decode_lime_batch(x, feature_names, encoders)
            return np.asarray(model.predict(df)).reshape(-1)

        if mode == "classification":
            exp = explainer.explain_instance(
                row_np,
                predict_proba_np,
                num_features=min(num_features, len(feature_names)),
            )
        else:
            exp = explainer.explain_instance(
                row_np,
                predict_np,
                num_features=min(num_features, len(feature_names)),
            )

        fig = exp.as_pyplot_figure()
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue(), None
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return b"", str(exc)
