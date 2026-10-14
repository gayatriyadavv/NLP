"""
Paired bootstrap confidence intervals for comparing two system configurations evaluated on the
same test items (figures / queries).

Resampling unit = test figure. Each resample draws N figures with replacement and recomputes the
metric for both systems on the same draw, so the CI on the difference accounts for the pairing.
"""

from typing import Callable, Dict, Sequence

import numpy as np


def paired_bootstrap(stat: Callable[[np.ndarray], float], n_items: int, n_resamples: int = 10000,
                     seed: int = 0) -> np.ndarray:
    """Returns ``n_resamples`` values of ``stat(idx)`` for bootstrap index draws ``idx``."""
    rng = np.random.default_rng(seed)
    return np.array([stat(rng.integers(0, n_items, n_items)) for _ in range(n_resamples)])


def summarize_difference(diffs: np.ndarray, observed: float) -> Dict[str, float]:
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: difference = 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"difference": round(float(observed), 4), "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4), "p_value_bootstrap": round(float(min(p, 1.0)), 4),
            "significant_at_0.05": bool(lo > 0 or hi < 0)}


def compare_means(a: Sequence[float], b: Sequence[float], n_resamples: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Paired difference of means, b - a (e.g. per-query reciprocal rank or per-figure SCQ)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert len(a) == len(b)
    d = b - a
    diffs = paired_bootstrap(lambda idx: d[idx].mean(), len(d), n_resamples, seed)
    return summarize_difference(diffs, d.mean())


def compare_micro_f1(counts_a: np.ndarray, counts_b: np.ndarray, n_resamples: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """
    Paired difference of micro-F1, b - a. ``counts_*`` are [N_figures, 3] arrays of (tp, fp, fn)
    per figure.
    """
    counts_a, counts_b = np.asarray(counts_a, float), np.asarray(counts_b, float)

    def f1(c):
        tp, fp, fn = c.sum(0)
        return 2 * tp / max(2 * tp + fp + fn, 1e-9)

    diffs = paired_bootstrap(lambda idx: f1(counts_b[idx]) - f1(counts_a[idx]), len(counts_a), n_resamples, seed)
    return summarize_difference(diffs, f1(counts_b) - f1(counts_a))
