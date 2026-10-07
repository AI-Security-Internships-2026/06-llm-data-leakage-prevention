"""experiments/analysis/stats.py -- Issue #10 statistics (no paper numbers typed by hand)."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def cluster_bootstrap_ci(values_by_cluster: dict, stat=np.mean, n_boot=5000, seed=0, alpha=0.05):
    """Percentile CI resampling *clusters* (seeds / victim sets), then items within.
    values_by_cluster: {cluster_id: [values]}.  Respects the hierarchy so that
    victims from one seed are not treated as independent of that seed."""
    rng = np.random.default_rng(seed)
    keys = list(values_by_cluster)
    arrs = [np.asarray(values_by_cluster[k], float) for k in keys]
    if not arrs or sum(len(a) for a in arrs) == 0:
        return (float("nan"),) * 3
    point = stat(np.concatenate(arrs))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(arrs), len(arrs))
        boots[b] = stat(np.concatenate([rng.choice(arrs[i], len(arrs[i])) for i in pick]))
    return (float(point), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def bootstrap_ci(x, stat=np.mean, n_boot=5000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = np.array([stat(rng.choice(x, x.size)) for _ in range(n_boot)])
    return (float(stat(x)), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2)))


def fisher_exact(k1, n1, k2, n2):
    """Two-sided Fisher exact test on two proportions -> (odds_ratio, p)."""
    o, p = st.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return float(o), float(p)


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def risk_difference_ci(k1, n1, k2, n2, z=1.96):
    """Newcombe hybrid-score CI for p1-p2 (good for small n / boundary p)."""
    l1, u1 = wilson_ci(k1, n1, z); l2, u2 = wilson_ci(k2, n2, z)
    d = k1 / n1 - k2 / n2
    return (float(d), float(d - np.sqrt((k1/n1 - l1)**2 + (u2 - k2/n2)**2)),
            float(d + np.sqrt((u1 - k1/n1)**2 + (k2/n2 - l2)**2)))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    u, _ = st.mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (a.size * b.size) - 1)


def mannwhitney_p(a, b) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    try:
        return float(st.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def roc_auc(pos_scores, neg_scores, higher_is_positive: bool = True) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 P(tie).  For the timing oracle a
    cache HIT means LOW ttft, so pass higher_is_positive=False."""
    p, n = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    if not higher_is_positive:
        p, n = -p, -n
    r = st.rankdata(np.concatenate([p, n]))
    return float((r[:p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size))


def tpr_at_fpr(pos_scores, neg_scores, fpr: float = 0.01, higher_is_positive: bool = False) -> float:
    p, n = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if p.size == 0 or n.size == 0:
        return float("nan")
    if not higher_is_positive:
        p, n = -p, -n
    thr = np.quantile(n, 1 - fpr, method="higher")
    return float(np.mean(p > thr))


def quantiles(x, qs=(50, 95, 99)):
    x = np.asarray(x, float)
    return {f"p{q}": (float(np.percentile(x, q)) if x.size else float("nan")) for q in qs}
