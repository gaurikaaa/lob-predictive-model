"""
Load the feature/label CSVs written by data_prep/build_features.py into plain
(X, y) NumPy arrays for the models in model.py.

Each CSV row is ONE order-book snapshot:

    col 0, 1                metadata (symbol, timestamp) -- ignored
    then pipe-delimited floats:
        [0 : data_dim]              44 LOB features (col 0 = mid_price, which
                                     is also the label's reference price)
        [data_dim]                  sigma (trailing volatility)
        [data_dim+1 : +label_dim]   forward VWAP at each of 12 horizons

One snapshot -> one training example: 44 features in, one price-change out.
The label at horizon `timestep` is turned into a sigma-normalized log return:

    y = log(vwap / mid_price) / sigma

so it is comparable across symbols and market regimes. Features are z-score
normalized per file.
"""
import csv
import os

import numpy as np

HORIZONS_SECONDS = [1, 2, 3, 5, 10, 20, 30, 60, 90, 120, 180, 300]


def _read_rows(path):
    """Read one feature CSV into an (n_snapshots, n_values) float array."""
    rows = []
    with open(path) as csvfile:
        for row in csv.reader(csvfile):
            values = []
            for idx, item in enumerate(row):
                if idx >= 2:  # skip the symbol/timestamp metadata columns
                    values.extend(float(v) for v in item.split('|'))
            rows.append(values)
    return np.array(rows)


def load_xy(root, list_file, timestep,
            data_dim=44, label_dim=len(HORIZONS_SECONDS), normalize=True):
    """Return (X, y) for every CSV named in `list_file`.

    X: (n, data_dim) features. y: (n,) sigma-normalized log return at horizon
    `timestep` (0 = 1s ... 11 = 300s).
    """
    with open(os.path.join(root, list_file)) as f:
        file_names = [line.strip() for line in f if line.strip()]

    x_parts, y_parts = [], []
    for name in file_names:
        raw = _read_rows(os.path.join(root, name))
        expected = data_dim + 1 + label_dim
        assert raw.shape[1] == expected, (
            f"{name}: expected {expected} columns "
            f"(data_dim={data_dim} + sigma + label_dim={label_dim}), got {raw.shape[1]}")

        features = raw[:, :data_dim]
        ref_price = raw[:, 0]
        sigma = raw[:, data_dim]
        vwap = raw[:, data_dim + 1 + timestep]
        label = np.log(vwap / ref_price) / (sigma + 1e-8)

        if normalize:
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            features = (features - mean) / (std + 1e-20)

        x_parts.append(features)
        y_parts.append(label)

    X = np.concatenate(x_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)
    return X, y
