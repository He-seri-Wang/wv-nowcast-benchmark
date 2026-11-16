"""Evaluation protocol shared by the pilots: window-RMS aggregation and
day-block bootstrap significance.

Rows everywhere are (day_str, rmse_of_one_window) pairs; the day key drives the
block bootstrap so windows from the same day are resampled together.
"""
import numpy as np


def agg(rows):
    """Aggregate per-window RMSE as an RMS over windows (in Kelvin)."""
    return float(np.sqrt(np.mean([r[1] ** 2 for r in rows])))


def day_boot(rows_a, rows_b, nb=1000, seed=0):
    """Standard deviation of agg(rows_a) - agg(rows_b) under day-block bootstrap.

    Significance convention across the benchmark: a model beats a baseline when
    its gain >= 2 x this se.
    """
    rng = np.random.default_rng(seed)
    days = sorted(set(r[0] for r in rows_a))
    ba = {d: [r for r in rows_a if r[0] == d] for d in days}
    bb = {d: [r for r in rows_b if r[0] == d] for d in days}
    ds = []
    for _ in range(nb):
        sel = rng.choice(days, len(days), replace=True)
        A = np.sqrt(np.mean([r[1] ** 2 for d in sel for r in ba[d]]))
        B = np.sqrt(np.mean([r[1] ** 2 for d in sel for r in bb[d]]))
        ds.append(A - B)
    return float(np.std(ds))
