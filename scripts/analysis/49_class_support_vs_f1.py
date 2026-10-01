#!/usr/bin/env python3
"""Per-class F1 against the number of training BioProjects and runs, for every model.

Question (2026-09-29): how many BioProjects and runs must support a class before the
models predict it with F1 >= 0.5?

Two sets, kept apart:

* dev folds: pooled out-of-fold predictions of the read-9 recipe. DIANA single-task
  arms at their epoch budgets (results/epoch_budget_v9/oof_<task>_<task>.tsv) and the
  four tuned baselines at their selected fraction configuration on the full v9 grid
  (results/baselines_sig_v9/pred_fraction_<task>_<model>.tsv). Guidance only.
* held-out, read 9: a re-scoring of predictions already taken
  (results/final_eval_budget_v9/heldout_<task>/test_predictions.tsv and
  results/baseline_predictions_v9/heldout_predictions.tsv). Not a new read. A threshold
  chosen from these numbers would be a decision shaped by held-out data; choose on the
  dev folds.

Per-class F1 is one-vs-rest F1 of the multiclass prediction. On held-out, runs whose
class never occurs in training are excluded as out-of-vocabulary, as read 9 did.
Chance level per class: a predictor answering the class at random with its own
prevalence p has precision = recall = p, so F1 = p (`chance_f1`).

Outputs, results/class_support_v9/:
  per_class_f1.tsv      one row per eligible class x model x set
  support_summary.tsv   n classes and share with F1 >= 0.5, by training-BioProject bin
                        and training-run bin, per model and set
  threshold_scan.tsv    for each minimum number of training BioProjects: classes kept,
                        mean per-class F1 and share >= 0.5, per model and set
  f1_vs_projects_<set>.png, f1_vs_runs_<set>.png

    ./env/bin/python scripts/analysis/49_class_support_vs_f1.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/class_support_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
BASELINES = ["LogisticRegression_Bal", "LinearSVM_Bal", "RandomForest_Bal", "kNN"]
MODELS = ["DIANA single-task"] + BASELINES
THRESHOLD = 0.5
PROJECT_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10 ** 6, "6+")]
RUN_BINS = [(0, 20, "<20"), (20, 100, "20-100"), (100, 10 ** 9, ">100")]
# colour = model family (DIANA vs tuned baseline; dataviz reference slots 1 and 2, validated
# all-pairs), marker = model. Five distinct hues fail the all-pairs floors for a scatter.
STYLE = {"DIANA single-task": ("#2a78d6", "o"), "LogisticRegression_Bal": ("#eb6834", "s"),
         "LinearSVM_Bal": ("#eb6834", "^"), "RandomForest_Bal": ("#eb6834", "D"), "kNN": ("#eb6834", "v")}


def per_class_f1(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, float]:
    return {c: float(f1_score(y == c, p == c, zero_division=0)) if (y == c).any() else np.nan for c in classes}


def dev_predictions(task: str, sel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pooled out-of-fold predictions per model: DataFrame indexed by run with y_true, y_pred."""
    out = {}
    d = pd.read_csv(ROOT / f"results/epoch_budget_v9/oof_{task}_{task}.tsv", sep="\t")
    d = d[d[f"{task}_true"].notna()]
    out["DIANA single-task"] = pd.DataFrame({"y_true": d[f"{task}_true"].astype(str).to_numpy(),
                                             "y_pred": d[f"{task}_pred"].astype(str).to_numpy()}, index=d.Run_accession)
    for m in BASELINES:
        cfg = sel[(sel.task == task) & (sel.model == m) & (sel.rep == "fraction")].cfg.iloc[0]
        b = pd.read_csv(ROOT / f"results/baselines_sig_v9/pred_fraction_{task}_{m}.tsv", sep="\t")
        b = b[b.cfg == cfg]
        out[m] = pd.DataFrame({"y_true": b.y_true.astype(str).to_numpy(), "y_pred": b.y_pred.astype(str).to_numpy()},
                              index=b.Run_accession)
    return out


def heldout_predictions(task: str, vocab: set, base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Read-9 held-out predictions per model, out-of-vocabulary runs excluded."""
    out = {}
    d = pd.read_csv(ROOT / f"results/final_eval_budget_v9/heldout_{task}/test_predictions.tsv", sep="\t")
    d = d[d[f"{task}_true"].astype(str).isin(vocab)]
    out["DIANA single-task"] = pd.DataFrame({"y_true": d[f"{task}_true"].astype(str).to_numpy(),
                                             "y_pred": d[f"{task}_pred"].astype(str).to_numpy()}, index=d.Run_accession)
    for m in BASELINES:
        b = base[(base.task == task) & (base.model == m)]
        b = b[b.y_true.astype(str).isin(vocab)]
        out[m] = pd.DataFrame({"y_true": b.y_true.astype(str).to_numpy(), "y_pred": b.y_pred.astype(str).to_numpy()},
                              index=b.Run_accession)
    return out


def label_bin(v: float, bins: list[tuple]) -> str:
    for lo, hi, name in bins:
        if (lo <= v <= hi) if bins is PROJECT_BINS else (lo < v <= hi):
            return name
    return "none"


def scatter(df: pd.DataFrame, x: str, xlabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(MODELS):
        s = df[(df.model == m) & df.f1.notna()]
        colour, marker = STYLE[m]
        jitter = np.exp(rng.uniform(-0.06, 0.06, len(s)) + (i - 2) * 0.05)
        ax.scatter(s[x] * jitter, s.f1, s=28, c=colour, marker=marker, alpha=0.85, linewidths=0.4,
                   edgecolors="white", label=m)
    ax.axhline(THRESHOLD, color="#52514e", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("per-class F1 (one-vs-rest)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, which="major", color="#e6e5e2", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    test = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", low_memory=False)
    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    sel = pd.read_csv(ROOT / "results/baselines_sig_v9/selected_configs_sig_vs_fraction.tsv", sep="\t")
    base_ho = pd.read_csv(ROOT / "results/baseline_predictions_v9/heldout_predictions.tsv", sep="\t")

    rows = []
    for task in TASKS:
        classes = sorted(elig[(elig.target == task) & elig.evaluable]["class"].astype(str))
        vocab = set(train[task].dropna().astype(str))
        tr, te = train[train[task].notna()], test[test[task].notna()]
        support = {c: {"n_train_runs": int((tr[task].astype(str) == c).sum()),
                       "n_train_projects": int(tr.loc[tr[task].astype(str) == c, "archive_project"].nunique()),
                       "n_heldout_runs": int((te[task].astype(str) == c).sum()),
                       "n_heldout_projects": int(te.loc[te[task].astype(str) == c, "archive_project"].nunique())}
                   for c in classes}
        for set_name, preds in (("dev-folds", dev_predictions(task, sel)),
                                ("held-out read 9", heldout_predictions(task, vocab, base_ho))):
            for m, d in preds.items():
                y, p = d.y_true.to_numpy(), d.y_pred.to_numpy()
                f1s = per_class_f1(y, p, classes)
                prev = pd.Series(y).value_counts(normalize=True)
                for c in classes:
                    rows.append({"set": set_name, "task": task, "class": c, "model": m, **support[c],
                                 "n_scored_runs": int((y == c).sum()), "chance_f1": float(prev.get(c, 0.0)),
                                 "f1": f1s[c]})
    df = pd.DataFrame(rows)
    df["project_bin"] = df.n_train_projects.map(lambda v: label_bin(v, PROJECT_BINS))
    df["run_bin"] = df.n_train_runs.map(lambda v: label_bin(v, RUN_BINS))
    df.to_csv(OUT / "per_class_f1.tsv", sep="\t", index=False)

    scored = df[df.f1.notna()].copy()
    scored["above"] = scored.f1 >= THRESHOLD
    summary = (scored.groupby(["set", "model", "project_bin", "run_bin"], observed=True)
               .agg(n_classes=("class", "size"), share_f1_ge_0_5=("above", "mean"), mean_f1=("f1", "mean"))
               .reset_index())
    summary.to_csv(OUT / "support_summary.tsv", sep="\t", index=False)

    scan = []
    for set_name, s in scored.groupby("set"):
        for k in range(1, 12):
            kept = s[s.n_train_projects >= k]
            for m, sm in kept.groupby("model"):
                scan.append({"set": set_name, "min_train_projects": k, "model": m, "n_classes": len(sm),
                             "mean_f1": sm.f1.mean(), "share_f1_ge_0_5": sm.above.mean()})
    scan = pd.DataFrame(scan)
    scan.to_csv(OUT / "threshold_scan.tsv", sep="\t", index=False)

    for set_name, tag in (("dev-folds", "devfolds"), ("held-out read 9", "heldout_read9")):
        s = scored[scored.set == set_name]
        scatter(s, "n_train_projects", "training BioProjects supporting the class",
                f"Per-class F1 by training BioProjects, {set_name}", OUT / f"f1_vs_projects_{tag}.png")
        scatter(s, "n_train_runs", "training runs of the class",
                f"Per-class F1 by training runs, {set_name}", OUT / f"f1_vs_runs_{tag}.png")

    pd.set_option("display.width", 220)
    piv = scored.pivot_table(index=["set", "project_bin"], columns="model", values="above", aggfunc="mean")
    n = scored[scored.model == MODELS[0]].groupby(["set", "project_bin"], observed=True)["class"].size()
    piv.insert(0, "n_classes", n)
    print("\nshare of classes with F1 >= 0.5, by training BioProjects:")
    print(piv.reindex(columns=["n_classes"] + MODELS).round(2).to_string())
    piv = scored.pivot_table(index=["set", "run_bin"], columns="model", values="above", aggfunc="mean")
    n = scored[scored.model == MODELS[0]].groupby(["set", "run_bin"], observed=True)["class"].size()
    piv.insert(0, "n_classes", n)
    print("\nshare of classes with F1 >= 0.5, by training runs:")
    print(piv.reindex(columns=["n_classes"] + MODELS).round(2).to_string())
    print("\nthreshold scan, dev folds (classes kept at >= k training BioProjects; mean per-class F1):")
    print(scan[scan.set == "dev-folds"].pivot_table(index="min_train_projects", columns="model", values="mean_f1")
          .join(scan[(scan.set == "dev-folds") & (scan.model == MODELS[0])].set_index("min_train_projects").n_classes)
          .reindex(columns=["n_classes"] + MODELS).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
