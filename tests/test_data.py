"""Tests for src/data.py on a tiny fake dataset."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_cache, load_data  # noqa: E402


def write_fake_dataset(folder, y_ids=None):
    train_ids = np.array([10, 11, 12, 13])
    test_ids = np.array([20, 21])
    if y_ids is None:
        y_ids = train_ids
    header = "Id,_STATE,GENHLTH,_BMI5\n"
    with open(os.path.join(folder, "x_train.csv"), "w") as f:
        f.write(header + "10,1,2,2500\n11,1,,3100\n12,9,7,\n13,4,1,1900\n")
    with open(os.path.join(folder, "x_test.csv"), "w") as f:
        f.write(header + "20,5,3,2200\n21,1,9,2800\n")
    with open(os.path.join(folder, "y_train.csv"), "w") as f:
        f.write("Id,_MICHD\n")
        for i, label in zip(y_ids, [-1, 1, -1, -1]):
            f.write(f"{i},{label}\n")
    return train_ids, test_ids


def test_cache_round_trip(tmp_path):
    train_ids, test_ids = write_fake_dataset(tmp_path)
    cache = tmp_path / "build" / "data.npz"
    x_train, x_test, y_train, tr_ids, te_ids, names = load_data(cache, tmp_path)
    assert cache.exists()
    assert x_train.shape == (4, 3) and x_test.shape == (2, 3)
    assert np.isnan(x_train[1, 1]) and np.isnan(x_train[2, 2])
    np.testing.assert_array_equal(y_train, [-1, 1, -1, -1])
    np.testing.assert_array_equal(tr_ids, train_ids)
    np.testing.assert_array_equal(te_ids, test_ids)
    assert list(names) == ["_STATE", "GENHLTH", "_BMI5"]

    # second call: reads the cache, the CSVs are no longer needed
    for name in ("x_train.csv", "x_test.csv", "y_train.csv"):
        os.remove(tmp_path / name)
    x_train2 = load_data(cache, tmp_path)[0]
    np.testing.assert_array_equal(x_train2, x_train)


def test_label_order_is_checked(tmp_path):
    write_fake_dataset(tmp_path, y_ids=np.array([11, 10, 12, 13]))
    with pytest.raises(ValueError, match="same order"):
        build_cache(tmp_path, tmp_path / "data.npz")
