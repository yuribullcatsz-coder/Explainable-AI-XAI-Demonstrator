"""Data preparation and model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainResult:
    """Outcome of `train_model`."""

    model: Pipeline
    task: str  # "classification" | "regression"
    feature_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def infer_task(y: pd.Series) -> str:
    """Infer classification vs regression from the target column."""
    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "classification"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    nunique = int(y.nunique(dropna=True))
    if nunique <= 10:
        return "classification"
    return "regression"


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Split X/y; categorical columns are user-selected, rest are numeric."""
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not in dataframe.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    all_cols = list(X.columns)
    if not all_cols:
        raise ValueError("No feature columns: only target column present.")

    cat_set = set(categorical_cols or [])
    unknown = cat_set - set(all_cols)
    if unknown:
        raise KeyError(f"Categorical columns not found in data: {sorted(unknown)}")

    categorical = [c for c in all_cols if c in cat_set]
    numeric = [c for c in all_cols if c not in cat_set]

    return X, y, numeric, categorical


def build_preprocess_and_model(
    numeric_cols: list[str],
    categorical_cols: list[str],
    task: str,
    random_state: int = 42,
) -> Pipeline:
    """Sklearn Pipeline: preprocess + RandomForest."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    if not transformers:
        raise ValueError("No feature columns after excluding target.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if task == "classification":
        estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=random_state,
        )

    return Pipeline(
        steps=[("preprocess", preprocessor), ("model", estimator)]
    )


def get_feature_names_after_preprocess(model: Pipeline) -> list[str]:
    """Feature names after ColumnTransformer + OneHot (post-fit)."""
    preprocess: ColumnTransformer = model.named_steps["preprocess"]
    names: list[str] = []
    for name, trans, cols in preprocess.transformers_:
        if name == "num":
            names.extend([str(c) for c in cols])
        elif name == "cat":
            ohe: OneHotEncoder = trans.named_steps["onehot"]
            for col, cats in zip(cols, ohe.categories_):
                for cat in cats:
                    names.append(f"{col}={cat}")
    return names


def train_model(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[TrainResult, Any, Any, Any, Any]:
    """
    Train/test split, fit pipeline, return (result, X_train, X_test, y_train, y_test).

    Rows with NaN in the target are dropped.
    """
    work = df.copy()
    work = work.dropna(subset=[target_col])
    if len(work) < 5:
        raise ValueError("Need at least 5 rows with non-null target values.")

    X, y_raw, numeric_cols, categorical_cols_list = prepare_features(
        work, target_col, categorical_cols
    )
    task = infer_task(y_raw)

    if task == "classification":
        y = y_raw.astype(str)
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            raise ValueError("Regression target contains non-numeric values.")
        y = y.astype(float)

    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if len(X_train) < 2 or len(X_test) < 1:
        raise ValueError("Not enough rows after train/test split.")

    pipeline = build_preprocess_and_model(
        numeric_cols, categorical_cols_list, task, random_state=random_state
    )
    pipeline.fit(X_train, y_train)

    feature_names = get_feature_names_after_preprocess(pipeline)

    result = TrainResult(
        model=pipeline,
        task=task,
        feature_names=feature_names,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols_list,
    )
    return result, X_train, X_test, y_train, y_test


def model_matrix_for_explain(
    model: Pipeline, X: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Transformed design matrix (for SHAP/LIME) and feature names."""
    preprocess = model.named_steps["preprocess"]
    Xm = preprocess.transform(X)
    names = get_feature_names_after_preprocess(model)
    return np.asarray(Xm), list(names)
