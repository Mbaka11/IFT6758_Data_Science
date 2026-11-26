import csv
import os
import random

import numpy as np
import torch


class RunningMeanStd(object):
    """Tracks the running mean and variance of a stream of data (parallel algorithm)."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total


def explained_variance(y_pred, y_true):
    """
    1 - Var[y_true - y_pred] / Var[y_true].
    1 means the value function predicts the returns perfectly, 0 means it is no better than
    predicting a constant, and a negative value means it is worse than a constant.
    """
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / var_y)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small networks run fastest on one CPU thread; more threads only add overhead.
    torch.set_num_threads(1)


class CSVLogger(object):
    """Writes one row per training iteration to <logdir>/log.csv and keeps the history in memory."""

    def __init__(self, logdir):
        self.path = os.path.join(logdir, "log.csv")
        self.rows = []
        self.keys = None

    def log(self, row):
        if self.keys is None:
            self.keys = list(row.keys())
        self.rows.append(row)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

    def history(self):
        return {k: [r.get(k) for r in self.rows] for k in (self.keys or [])}
