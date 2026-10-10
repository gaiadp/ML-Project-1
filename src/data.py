"""Load the dataset from the CSV files once and cache it in build/data.npz.

Build the cache with the Run button on this file, or:  python src/data.py
"""

import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repository root on the import path, so that the Run button (python src/data.py) works
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers import load_csv_data  # noqa: E402

DATA_DIR = os.path.join(ROOT, "dataset")
CACHE_PATH = os.path.join(ROOT, "build", "data.npz")

KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids", "feature_names")


def read_feature_names(csv_path):
    """Return the column names of a data CSV without "Id", as a 1D array of str.

    feature_names[j] is the name of column j of the x returned by load_csv_data.
    """
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    return np.array(header[1:])


def build_cache(data_path=DATA_DIR, cache_path=CACHE_PATH):
    """Read the CSVs with load_csv_data and save all arrays to cache_path.

    Raises ValueError if x_train.csv and x_test.csv have different columns or if
    y_train.csv is not in the same row order as x_train.csv.

    Returns:
        dict of the saved arrays, with the keys in KEYS
    """
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    feature_names = read_feature_names(os.path.join(data_path, "x_train.csv"))
    test_names = read_feature_names(os.path.join(data_path, "x_test.csv"))
    if not np.array_equal(feature_names, test_names):
        raise ValueError("x_train.csv and x_test.csv have different columns")
    if len(feature_names) != x_train.shape[1]:
        raise ValueError("header length does not match the number of columns")

    # load_csv_data reads the labels without their Id: check the row order
    y_ids = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=0,
    )
    if not np.array_equal(y_ids, train_ids):
        raise ValueError("y_train.csv rows are not in the same order as x_train.csv")

    arrays = dict(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        train_ids=train_ids,
        test_ids=test_ids,
        feature_names=feature_names,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # uncompressed: larger on disk (~1.1 GB) but fast to reload
    np.savez(cache_path, **arrays)
    return arrays


def load_data(cache_path=CACHE_PATH, data_path=DATA_DIR):
    """Return (x_train, x_test, y_train, train_ids, test_ids, feature_names).

    Reads the cache, building it first from the CSVs if it does not exist.
    """
    if not os.path.exists(cache_path):
        build_cache(data_path, cache_path)
    with np.load(cache_path) as data:
        return tuple(data[key] for key in KEYS)


if __name__ == "__main__":
    start = time.time()
    build_cache()
    print(f"Cache built in {time.time() - start:.0f} s: {CACHE_PATH}")
    print(f"Size on disk: {os.path.getsize(CACHE_PATH) / 1e9:.2f} GB")

    start = time.time()
    x_train, x_test, y_train, train_ids, test_ids, names = load_data()
    print(f"Reloaded with np.load in {time.time() - start:.1f} s")
    print(f"x_train {x_train.shape} {x_train.dtype}, x_test {x_test.shape}")
    print(f"y_train {y_train.shape}, labels {np.unique(y_train)}")
    print(f"positives: {np.mean(y_train == 1):.2%}")
    print(f"test ids: {test_ids[:3]} ... {test_ids[-1]}")
    print(f"first columns: {names[:5]}")
