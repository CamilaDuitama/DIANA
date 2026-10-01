#!/usr/bin/env python3
"""Are shallow samples misclassified more often than deep ones, within the same class?

Cheap diagnostic on predictions that already exist: the pooled dev-fold out-of-fold
predictions of DIANA single-task at the read-9 budgets (results/epoch_budget_v9/oof_*)
and of the tuned logistic regression (results/baselines_sig_v9/pred_fraction_*, selected
config). Depth proxy: the number of non-zero unitigs of the run (nonzero_counts.tsv), on a
log10 scale.

Per task and model:
  * per eligible class with at least 20 labelled out-of-fold runs: accuracy on the
    shallow half (below the class median depth) and on the deep half, and their difference;
  * pooled within-class effect: a logistic regression of correctness on log10 depth with
    class fixed effects; the slope is the log-odds change per tenfold depth, with a
    bootstrap interval over BioProjects (500 resamples).
Plots, one per task and model (results/depth_vs_correctness_v9/): the shallow-half and
deep-half accuracy per class.

    ./env/bin/python scripts/analysis/56_depth_vs_correctness.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPLITS, OUT = ROOT / "data/splits_v9", ROOT / "results/depth_vs_correctness_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
MIN_RUNS, N_BOOT = 20, 500
TRAIN_C, HELD_C, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e2"   # validated slots 1 and 2


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def predictions(task: str) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(ROOT / f"results/epoch_budget_v9/oof_{task}_{task}.tsv", sep="\t").dropna(subset=[f"{task}_true"])
    out = {"DIANA single-task": pd.DataFrame({"y": d[f"{task}_true"].astype(str).to_numpy(),
                                              "p": d[f"{task}_pred"].astype(str).to_numpy()}, index=d.Run_accession.to_numpy())}
    sel = pd.read_csv(ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv", sep="\t")
    cfg = sel[(sel.task == task) & (sel.model == "LogisticRegression_Bal") & (sel.rep == "fraction")].cfg.iloc[0]
    b = pd.read_csv(ROOT / f"results/baselines_sig_v9/pred_fraction_{task}_LogisticRegression_Bal.tsv", sep="\t")
    b = b[b.cfg == cfg]
    out["LogisticRegression_Bal"] = pd.DataFrame({"y": b.y_true.astype(str).to_numpy(), "p": b.y_pred.astype(str).to_numpy()},
                                                 index=b.Run_accession.to_numpy())
    return out


def slope_with_ci(df: pd.DataFrame, key) -> tuple[float, float, float]:
    """Log-odds of correctness per log10 depth, class fixed effects, bootstrap over projects."""
    def fit(sub: pd.DataFrame) -> float:
        if sub.correct.nunique() < 2 or sub["class"].nunique() < 1:
            return np.nan
        Xd = pd.get_dummies(sub["class"], drop_first=False).astype(float)
        X = np.column_stack([sub.log_depth.to_numpy() - sub.log_depth.mean(), Xd.to_numpy()])
        m = LogisticRegression(C=1e6, fit_intercept=False, max_iter=2000).fit(X, sub.correct.astype(int))
        return float(m.coef_[0][0])
    obs = fit(df)
    uniq = df.project.unique(); by = {q: df.index[df.project == q] for q in uniq}
    rng = rng_for(*key); vals = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[q] for q in rng.choice(uniq, size=len(uniq), replace=True)])
        vals.append(fit(df.loc[sel]))
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
    return obs, lo, hi


def plot(per_class: pd.DataFrame, task: str, model: str) -> None:
    d = per_class.sort_values("n_runs", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(d) + 2), 4.2))
    x = np.arange(len(d))
    for i in x:
        ax.plot([i, i], [d.acc_shallow.iloc[i], d.acc_deep.iloc[i]], color=MUTED, lw=1, zorder=1)
    ax.scatter(x, d.acc_shallow, color=HELD_C, s=46, zorder=2, label="shallow half (below class median depth)")
    ax.scatter(x, d.acc_deep, color=TRAIN_C, s=46, marker="s", zorder=2, label="deep half")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(d["class"], d.n_runs)], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(-0.03, 1.03); ax.set_ylabel("out-of-fold accuracy within class")
    ax.set_title(f"Accuracy by depth within class: {task}, {model}")
    ax.grid(True, axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"acc_by_depth_{task}_{model.replace(' ', '_')}.png", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    depth = pd.read_csv(OUT / "nonzero_counts.tsv", sep="\t").set_index("Run_accession")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t"); elig = elig[elig.evaluable]
    per_class_rows, pooled_rows = [], []
    for task in TASKS:
        eligible = set(elig[elig.target == task]["class"].astype(str))
        for model, pr in predictions(task).items():
            df = pr.join(depth[["n_nonzero"]]).join(meta[["archive_project"]]).rename(columns={"archive_project": "project"})
            df = df[df.y.isin(eligible) & (df.n_nonzero > 0)].copy()
            df["correct"] = (df.y == df.p).astype(int); df["log_depth"] = np.log10(df.n_nonzero); df["class"] = df.y
            for c, g in df.groupby("class"):
                if len(g) < MIN_RUNS:
                    continue
                med = g.log_depth.median()
                sh, dp = g[g.log_depth <= med], g[g.log_depth > med]
                per_class_rows.append({"task": task, "model": model, "class": c, "n_runs": len(g), "n_projects": g.project.nunique(),
                                       "median_nonzero": int(10 ** med), "acc_shallow": sh.correct.mean(), "acc_deep": dp.correct.mean(),
                                       "deep_minus_shallow": dp.correct.mean() - sh.correct.mean()})
            big = df[df["class"].map(df["class"].value_counts()) >= MIN_RUNS]
            obs, lo, hi = slope_with_ci(big.reset_index(drop=True), (task, model))
            pooled_rows.append({"task": task, "model": model, "n_runs": len(big), "n_classes": big["class"].nunique(),
                                "logodds_per_log10_depth": obs, "ci_low": lo, "ci_high": hi,
                                "accuracy_shallow_half": big[big.log_depth <= big.groupby("class").log_depth.transform("median")].correct.mean(),
                                "accuracy_deep_half": big[big.log_depth > big.groupby("class").log_depth.transform("median")].correct.mean()})
            pc = pd.DataFrame([r for r in per_class_rows if r["task"] == task and r["model"] == model])
            if len(pc):
                plot(pc, task, model)
    pc, pooled = pd.DataFrame(per_class_rows), pd.DataFrame(pooled_rows)
    pc.to_csv(OUT / "per_class_depth_accuracy.tsv", sep="\t", index=False)
    pooled.to_csv(OUT / "pooled_depth_effect.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("pooled within-class effect of depth on correctness (log-odds per tenfold increase in non-zero unitigs; CI over projects):")
    print(pooled.round(3).to_string(index=False))
    print("\nper class, shallow vs deep half (DIANA single-task):")
    print(pc[pc.model == "DIANA single-task"][["task", "class", "n_runs", "median_nonzero", "acc_shallow", "acc_deep", "deep_minus_shallow"]].round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
