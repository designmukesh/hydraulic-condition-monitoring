import numpy as np
from src.evaluation import split_indices, bootstrap_interval, score


def test_holdout_disjoint_reproducible_and_stratified():
    y = np.repeat([0, 1, 2], 100)
    train, test = split_indices(y)
    assert not set(train) & set(test)
    assert len(set(train) | set(test)) == 300
    assert np.array_equal(test, split_indices(y)[1])
    assert np.bincount(y[test]).tolist() == [20, 20, 20]


def test_fixed_label_metric_and_bootstrap():
    assert score([0, 0], [0, 0], [0, 1]) == .5
    y = np.array([0, 0, 1, 1])
    assert bootstrap_interval(y, y, [0, 1], 20) == [1, 1]
