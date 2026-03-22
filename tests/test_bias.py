import numpy as np
import pandas as pd
import pytest

from xai_demo.bias import compute_group_metrics


def test_bias_binary_groups():
    pred = np.array(["yes", "yes", "no", "no", "yes", "no"])
    sens = pd.Series(["A", "A", "A", "B", "B", "B"])
    rep = compute_group_metrics(pred, sens, positive_label="yes")
    assert rep.message is None
    assert rep.n_group_a == 3 and rep.n_group_b == 3
    assert rep.positive_rate_group_a == pytest.approx(2 / 3)
    assert rep.positive_rate_group_b == pytest.approx(1 / 3)


def test_bias_nonbinary_sensitive():
    pred = np.array([1, 0, 1])
    sens = pd.Series(["a", "b", "c"])
    rep = compute_group_metrics(pred, sens)
    assert rep.message is not None
