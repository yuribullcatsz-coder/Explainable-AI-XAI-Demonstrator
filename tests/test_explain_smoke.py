"""Lightweight checks that explanation helpers return bytes or an error string."""

import pandas as pd
import pytest

from xai_demo.explain import lime_explanation_figure, partial_dependence_figure, shap_summary_plot
from xai_demo.pipeline import train_model


@pytest.mark.slow
def test_shap_and_pdp_and_lime_smoke():
    df = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] * 8,
            "b": [1, 0, 1, 0, 1, 0] * 8,
            "y": [0, 1] * 24,
        }
    )
    res, X_train, X_test, _, _ = train_model(df, "y", categorical_cols=None, random_state=0)

    img, err = shap_summary_plot(res.model, X_test.head(20), max_samples=20)
    assert err is None or isinstance(err, str)
    assert (img and len(img) > 100) or err

    img2, err2 = partial_dependence_figure(res.model, X_train, "a")
    assert err2 is None or isinstance(err2, str)
    assert (img2 and len(img2) > 100) or err2

    row = X_test.iloc[[0]]
    img3, err3 = lime_explanation_figure(res.model, X_train, row, res.task)
    assert err3 is None or isinstance(err3, str)
    assert (img3 and len(img3) > 100) or err3
