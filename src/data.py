"""Load the dataset from the CSV files once, then reuse a NumPy cache in build/.

np.genfromtxt (used by helpers.load_csv_data) needs about a minute for x_train.csv;
np.load on the cached arrays takes a few seconds. build/ and dataset/ are gitignored.

Build the cache once, from the repository root:
    python -m src.data

Then, in any script run from the repository root (python run.py, python -m ...):
    from src.data import load_data
    x_train, x_test, y_train, train_ids, test_ids, feature_names = load_data()
"""

import csv
import os
import time

import numpy as np

from helpers import load_csv_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset")
CACHE_PATH = os.path.join(ROOT, "build", "data.npz")

# Order of the arrays returned by load_data (same as load_csv_data, plus the names)
KEYS = ("x_train", "x_test", "y_train", "train_ids", "test_ids", "feature_names")


def read_feature_names(csv_path):
    """Return the column names of a data CSV without "Id", as a 1D array of str.

    They are in the same order as the columns of the x returned by load_csv_data,
    which drops the Id column: feature_names[j] is the name of x[:, j].
    """
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f))
    return np.array(header[1:])


def build_cache(data_path=DATA_DIR, cache_path=CACHE_PATH):
    """Read the CSVs with load_csv_data and save all arrays to cache_path (.npz).

    Before saving, checks that x_train.csv and x_test.csv have the same columns and
    that the rows of y_train.csv are in the same order as those of x_train.csv
    (load_csv_data reads the labels without their Id column, so a different order
    would silently assign the wrong label to each person).

    Returns:
        the dict of arrays that was saved, with the keys in KEYS
    """
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)

    feature_names = read_feature_names(os.path.join(data_path, "x_train.csv"))
    test_names = read_feature_names(os.path.join(data_path, "x_test.csv"))
    if not np.array_equal(feature_names, test_names):
        raise ValueError("x_train.csv and x_test.csv have different columns")
    if len(feature_names) != x_train.shape[1]:
        raise ValueError("header length does not match the number of columns")

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
    # Uncompressed: about 1.1 GB on disk, but np.load just reads it back (fast)
    np.savez(cache_path, **arrays)
    return arrays


def load_data(cache_path=CACHE_PATH, data_path=DATA_DIR):
    """Return (x_train, x_test, y_train, train_ids, test_ids, feature_names).

    Reads the cache if it exists, otherwise builds it first from the CSVs (slow,
    only the first time). Labels are in {-1, 1}, as returned by load_csv_data.
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
