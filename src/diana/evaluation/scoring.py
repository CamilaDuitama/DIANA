"""One scoring definition for every DIANA comparison (PROTOCOLS.md, v19, C2).

Scored runs. A run counts for a task only when its true class has training rows in
the fit of the model being scored: on a dev fold, the other four folds; on the
held-out set, all training runs. A run of a class the model never saw cannot be
predicted by any classifier; it is counted and reported, never scored as an error.
Every model compared on a task is scored on the same runs.

Score. Macro-F1 over the eligible classes present among the scored runs. On the dev
folds the out-of-fold predictions of the five folds are pooled and scored once.

Intervals. BioProjects are resampled with replacement (2,000 draws) and every run of
a drawn project comes with it; the class set is recomputed on each resample. A paired
difference uses one set of draws for both models on an asserted-identical run set.

Regime. A class's training-run count: few < 20 <= medium < 100 <= many.

Epoch selection. Centred 5-epoch moving average of the criterion curve; the epoch
count is the argmax up to the minimum of the smoothed training loss; an argmax within
5 epochs of the last epoch run has not peaked.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

N_BOOT = 2000
WINDOW = 5
REGIMES = (("few", 0, 20), ("medium", 20, 100), ("many", 100, np.inf))


def regime_of(n_train_runs: float) -> str:
    """Regime of a class from its training-run count."""
    for name, lo, hi in REGIMES:
        if lo <= n_train_runs < hi:
            return name
    raise ValueError(f"no regime for {n_train_runs}")


def fit_classes_by_fold(labels: pd.Series, folds: pd.Series) -> dict[int, set[str]]:
    """For each dev fold k, the classes with at least one labelled run outside fold k."""
    if not labels.index.equals(folds.index):
        raise ValueError("labels and folds must share one run index")
    lab = labels.notna()
    return {int(k): set(labels[lab & (folds != k)].astype(str))
            for k in sorted(folds.unique())}


def scored_devfold(labels: pd.Series, folds: pd.Series) -> pd.Series:
    """True for runs whose label has fit rows when their own fold is held back."""
    fit = fit_classes_by_fold(labels, folds)
    ok = [pd.notna(v) and str(v) in fit[int(k)] for v, k in zip(labels, folds)]
    return pd.Series(ok, index=labels.index, name="scored")


def scored_heldout(labels: pd.Series, train_classes: Iterable[str]) -> pd.Series:
    """True for held-out runs whose label occurs among the training labels."""
    tc = set(map(str, train_classes))
    ok = [pd.notna(v) and str(v) in tc for v in labels]
    return pd.Series(ok, index=labels.index, name="scored")


def f1_macro_eligible(y: np.ndarray, pred: np.ndarray, eligible: set) -> float:
    """Macro-F1 over the eligible classes present in y; NaN when none is present."""
    y = np.asarray(y).astype(str)
    pred = np.asarray(pred).astype(str)
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    if not labs:
        return float("nan")
    return float(f1_score(y, pred, labels=labs, average="macro", zero_division=0))


def balanced_accuracy_eligible(y: np.ndarray, pred: np.ndarray, eligible: set) -> float:
    """Mean recall over the eligible classes present in y; NaN when none is present."""
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    if not labs:
        return float("nan")
    return float(np.mean([(pred[y == c] == c).mean() for c in labs]))


def accuracy_eligible(y: np.ndarray, pred: np.ndarray, eligible: set) -> float:
    """Share of correct predictions among the runs whose class is eligible; NaN when there are none."""
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    m = np.isin(y, sorted(eligible))
    return float((pred[m] == y[m]).mean()) if m.any() else float("nan")


METRICS = {"macro_f1": f1_macro_eligible, "balanced_accuracy": balanced_accuracy_eligible, "accuracy": accuracy_eligible}


def project_draws(groups: np.ndarray, key: str, n_boot: int = N_BOOT) -> list[np.ndarray]:
    """Row indices of n_boot BioProject resamples; the same key gives the same draws."""
    groups = np.asarray(groups).astype(str)
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    return [np.concatenate([by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(n_boot)]


def interval(values: list[float]) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def n_finite(values) -> int:
    """How many resamples gave a value (the rest are dropped by interval())."""
    return int(np.isfinite(np.asarray(values, float)).sum())


def score_with_interval(y, pred, groups, eligible: set, key: str, n_boot: int = N_BOOT, metric=f1_macro_eligible) -> dict:
    """Point score (macro-F1 unless `metric` is another METRICS function) and its BioProject percentile
    interval; the value is under "f1_macro_eligible" for macro-F1 and under "value" for any metric."""
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    draws = project_draws(groups, key, n_boot)
    lo, hi = interval([metric(y[i], pred[i], eligible) for i in draws])
    point = metric(y, pred, eligible)
    return {**({"f1_macro_eligible": point} if metric is f1_macro_eligible else {}), "value": point, "ci_low": lo, "ci_high": hi,
            "n_runs": int(len(y)), "n_projects": int(len(np.unique(np.asarray(groups)))),
            "n_classes": int(len({c for c in y.tolist() if c in eligible}))}


def paired_difference(runs_a: pd.Index, runs_b: pd.Index, y, pred_a, pred_b, groups,
                      eligible: set, key: str, n_boot: int = N_BOOT, metric=f1_macro_eligible) -> dict:
    """a minus b on identical runs and shared draws; verdict a / b / tie."""
    if not pd.Index(runs_a).equals(pd.Index(runs_b)):
        raise ValueError("paired comparison on different run sets")
    y = np.asarray(y).astype(str)
    pa, pb = np.asarray(pred_a).astype(str), np.asarray(pred_b).astype(str)
    obs = metric(y, pa, eligible) - metric(y, pb, eligible)
    d = [metric(y[i], pa[i], eligible) - metric(y[i], pb[i], eligible) for i in project_draws(groups, key, n_boot)]
    lo, hi = interval(d)
    v = np.asarray(d, float)
    v = v[np.isfinite(v)]
    return {"delta": float(obs), "ci_low": lo, "ci_high": hi, "p_boot": bootstrap_p(d),
            "frac_boot_positive": float((v > 0).mean()) if len(v) else float("nan"),
            "verdict": "a" if lo > 0 else ("b" if hi < 0 else "tie")}


def bootstrap_p(draws) -> float:
    """Two-sided bootstrap p-value of a paired difference from its resampled values:
    2 * min(#(d <= 0) + 1, #(d >= 0) + 1) / (n + 1), capped at 1; NaN draws are dropped."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).sum() + 1, (d >= 0).sum() + 1) / (len(d) + 1)))


def bh_adjust(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q) for one family of tests; NaN stays NaN."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if not len(ok):
        return q
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(adj[::-1])[::-1])
    return q


def smooth(curve: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Centred moving average; the ends average over the epochs available."""
    return pd.Series(np.asarray(curve, float)).rolling(window, center=True, min_periods=1).mean().to_numpy()


def select_epoch(criterion: np.ndarray, train_loss: np.ndarray, window: int = WINDOW) -> dict:
    """Epoch count from a pooled out-of-fold criterion curve (1-based epochs)."""
    crit = np.asarray(criterion, float)
    loss = np.asarray(train_loss, float)
    if crit.shape != loss.shape:
        raise ValueError(f"criterion has {crit.shape} epochs, training loss {loss.shape}")
    if not np.isfinite(crit).any():
        raise ValueError("criterion curve has no finite value")
    sm = smooth(crit, window)
    cap = int(np.argmin(smooth(loss, window))) + 1
    best = int(np.nanargmax(sm[:cap])) + 1
    uncapped = int(np.nanargmax(sm)) + 1
    n = len(crit)
    return {"epochs": best, "criterion_smoothed": float(sm[best - 1]), "criterion_raw": float(crit[best - 1]),
            "loss_cap_epoch": cap, "epochs_uncapped": uncapped, "capped": uncapped > cap,
            "peaked": best <= n - window, "n_epochs_run": n}


def eligible_classes(eligibility_tsv, task: str) -> set[str]:
    """Evaluable classes of a task from class_eligibility.tsv."""
    t = pd.read_csv(eligibility_tsv, sep="\t")
    if not {"target", "class", "evaluable"} <= set(t.columns):
        raise ValueError(f"{eligibility_tsv}: not an eligibility table")
    return set(t[(t.target == task) & t.evaluable]["class"].astype(str))


def check_same_split(meta: pd.DataFrame, expected_runs: Iterable[str], what: str) -> None:
    """Raise unless meta holds exactly the expected runs (catches mixed splits)."""
    got, want = set(meta["Run_accession"].astype(str)), set(map(str, expected_runs))
    if got != want:
        raise ValueError(f"{what}: {len(got - want)} unexpected and {len(want - got)} missing runs")


def regime_classes(train_labels: pd.Series, eligible: set) -> Mapping[str, set]:
    """Eligible classes grouped by regime of their training-run count."""
    counts = train_labels.dropna().astype(str).value_counts()
    out: dict[str, set] = {name: set() for name, _, _ in REGIMES}
    for c in eligible:
        out[regime_of(int(counts.get(c, 0)))].add(c)
    return out


# --- Mislabel flagging (C6) ------------------------------------------------------
#
# Flag score = 1 - P(stated label). A run enters the flagging pool for a task only if
# it had a label and its stated label (planted where planted, else the true one) is a
# class the model was fitted on; a label the model never saw cannot be assessed and is
# counted, not scored. Recall at a budget b is the recall of the last ROC operating
# point whose false-flag rate is <= b. A budget is attainable only if the share of
# correctly labelled runs tied at the maximum flag score is <= b ("can't check" if not).
# A paired comparison averages the budgets attainable by both models, fixed on the full
# data and reused on every resample.

BUDGETS = (0.05, 0.10, 0.20)


def recall_at_budget(score: np.ndarray, planted: np.ndarray, budget: float) -> float:
    from sklearn.metrics import roc_curve
    planted = np.asarray(planted, bool)
    if planted.all() or not planted.any():
        return float("nan")
    fpr, tpr, _ = roc_curve(planted, np.asarray(score, float), drop_intermediate=False)
    return float(tpr[max(int(np.searchsorted(fpr, budget, side="right")) - 1, 0)])


def saturated_share(score: np.ndarray, planted: np.ndarray) -> float:
    score, planted = np.asarray(score, float), np.asarray(planted, bool)
    clean = score[~planted]
    if not len(clean):
        return float("nan")
    return float((clean >= score.max() - 1e-9).mean())


def attainable_budgets(score: np.ndarray, planted: np.ndarray) -> tuple[float, ...]:
    sat = saturated_share(score, planted)
    return tuple(b for b in BUDGETS if np.isfinite(sat) and sat <= b)


def recall_if_attainable(score: np.ndarray, planted: np.ndarray, budget: float,
                         attainable: Iterable[float]) -> float:
    """Recall at a budget, NaN ("can't check") unless the budget is in `attainable`
    (decided once on the full data and reused on every resample)."""
    return recall_at_budget(score, planted, budget) if budget in tuple(attainable) else float("nan")


def choosing_score(score: np.ndarray, planted: np.ndarray, budgets: Iterable[float] | None = None) -> float:
    """Mean recall over the given budgets (default: the model's attainable budgets)."""
    budgets = attainable_budgets(score, planted) if budgets is None else tuple(budgets)
    if not budgets:
        return float("nan")
    return float(np.mean([recall_at_budget(score, planted, b) for b in budgets]))
