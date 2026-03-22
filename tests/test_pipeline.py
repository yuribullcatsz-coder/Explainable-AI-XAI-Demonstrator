import numpy as np
import pandas as pd
import pytest

from xai_demo.pipeline import (
    infer_task,
    prepare_features,
    train_model,
)


def test_infer_task_classification_strings():
    s = pd.Series(["a", "b", "a"])
    assert infer_task(s) == "classification"


def test_infer_task_regression_continuous():
    s = pd.Series(np.linspace(0, 1, 50))
    assert infer_task(s) == "regression"


def test_infer_task_small_int_is_classification():
    s = pd.Series([0, 1, 2] * 5)
    assert infer_task(s) == "classification"


def test_prepare_features_unknown_categorical():
    df = pd.DataFrame({"a": [1], "b": [2], "y": [0]})
    with pytest.raises(KeyError):
        prepare_features(df, "y", categorical_cols=["missing"])


def test_train_model_iris_like():
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map({0: "s0", 1: "s1", 2: "s2"})
    res, X_train, X_test, y_train, y_test = train_model(
        df, "species", categorical_cols=None, test_size=0.25, random_state=0
    )
    assert res.task == "classification"
    assert len(res.feature_names) > 0
    assert len(X_train) > 0 and len(X_test) > 0
    score = res.model.score(X_test, y_test)
    assert score > 0.8


def test_train_model_with_categorical_column():
    df = pd.DataFrame(
        {
            "color": ["r", "g", "b", "r", "g", "b"] * 5,
            "x": np.arange(30, dtype=float),
            "y": [0, 1] * 15,
        }
    )
    res, *_ = train_model(df, "y", categorical_cols=["color"], random_state=0)
    assert res.task == "classification"
    assert "color" in res.categorical_cols
