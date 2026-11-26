"""Statistics (plan §7.5): mean +- SD over repetitions, 95% CI, paired bootstrap, McNemar."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def mean_ci(values, confidence: float = 0.95) -> dict:
    values = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    mean, sd = float(values.mean()), float(values.std(ddof=1)) if n > 1 else 0.0
    half = float(st.t.ppf((1 + confidence) / 2, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half, "n": n}


def paired_bootstrap(y, pred_a, pred_b, metric, n_samples: int = 10000, seed: int = 0) -> dict:
    """Difference metric(a) - metric(b) on the same test ids; two-sided p-value of H0: no difference."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    observed = metric(y, a) - metric(y, b)
    diffs = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(observed), "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_value": float(min(p, 1.0))}


def mcnemar(y, pred_a, pred_b) -> dict:
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    a_ok, b_ok = a == y, b == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p = float(st.binomtest(min(n01, n10), n, 0.5).pvalue) if n < 50 else float(st.chi2.sf((abs(n01 - n10) - 1) ** 2 / n, 1))
    return {"n01": n01, "n10": n10, "p_value": p}
