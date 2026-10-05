#!/usr/bin/env python3
"""HELD-OUT READ 11: score the final DIANA, the tuned baselines and the DIANA + logistic
regression average on the 922 held-out runs, once. Run only after run_final_test_fits_v9.sbatch
and only with Camila's approval (PROJECT.md ledger).

Final DIANA = the 5-seed average of the reference recipe fitted on all training runs, with the
`feature` arm taken from the equal-study-weight fits (Phase 4 step 4.2, the one adopted change).
Baselines = results/baseline_predictions_v9/ (tuned, read 9): the probability files for the
detector-capable models, plus heldout_predictions.tsv for LinearSVM_Bal, which joins the
classification table only (no predict_proba, so no flag score; PROJECT.md §2).
Average = (P_DIANA + P_logistic) / 2 over the union of class columns, no tuning (X11 3.4).

Two scores, both with BioProject-level intervals (34 held-out projects) and paired differences
on shared project draws:
  classification  f1_macro_eligible (diana.evaluation.metrics), 2,000 resamples
  flagging        planted mislabels on the held-out metadata (results/planted_mislabels_heldout_v9/,
                  20 seeded single-label and 20 whole-record plantings, scores averaged over
                  plantings as on the dev folds): recall at 5 % (headline, last ROC point with a
                  false-flag rate <= 5 %), 10 %, 20 %, the choosing score and the can't-check
                  rule of X9.0, 1,000 resamples

    ./env/bin/python scripts/evaluation/20_final_test_score_v9.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
F = __import__("59_devfold_flagging_score")
from diana.evaluation.metrics import classification_metrics, load_eligible_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
SEEDS = [42, 1, 2, 3, 4]
BASELINES = ["LogisticRegression_Bal", "RandomForest_Bal", "kNN", "MajorityClass"]
N_BOOT_F1, N_BOOT_FLAG, SEED = 2000, 1000, 42


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def diana_probs(base: Path, task: str) -> pd.DataFrame:
    dirs = [base / task / f"seed{s}" for s in SEEDS]
    runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
    classes = json.load(open(dirs[0] / "label_classes.json"))[task]
    P = np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0)
    return pd.DataFrame(P, index=runs, columns=classes)


def baseline_probs(base: Path, task: str, model: str) -> pd.DataFrame:
    d = pd.read_csv(base / f"heldout_probabilities_{task}.tsv", sep="\t")
    d = d[d.model == model].set_index("Run_accession")
    cols = [c for c in d.columns if c.startswith("p_")]
    return d[cols].rename(columns={c: c[2:] for c in cols})


def average(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(a.columns) | set(b.columns)); idx = a.index.intersection(b.index)
    return (a.reindex(index=idx, columns=cols).fillna(0.0) + b.reindex(index=idx, columns=cols).fillna(0.0)) / 2.0


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diana", type=Path, default=ROOT / "results/final_test_v9/reference")
    ap.add_argument("--feature-from", type=Path, default=ROOT / "results/final_test_v9/studyweight", help="fits used for the feature arm (4.2); omit to use --diana for every task")
    ap.add_argument("--baselines", type=Path, default=ROOT / "results/baseline_predictions_v9")
    ap.add_argument("--planted-dir", type=Path, default=ROOT / "results/planted_mislabels_heldout_v9")
    ap.add_argument("--benchmarks", default="single=planted_test_mixed_r0.1_s*.tsv,record=planted_test_record_r0.1_s*.tsv",
                    help="name=glob pairs inside --planted-dir; scores are averaged over the matching plantings")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/splits_v9/test_metadata.tsv")
    ap.add_argument("--eligibility", type=Path, default=ROOT / "data/splits_v9/class_eligibility.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "results/final_test_v9/scores")
    a = ap.parse_args()
    if (a.out / "classification.tsv").exists():
        raise SystemExit(f"{a.out} already holds a read; refusing to score the held-out set twice")
    a.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(a.metadata, sep="\t", low_memory=False).set_index("Run_accession")
    import glob as _glob
    benches = {}
    for spec in a.benchmarks.split(","):
        name, pat = spec.split("=", 1)
        files = sorted(_glob.glob(str(a.planted_dir / pat)))
        if not files:
            raise SystemExit(f"no plantings match {pat} in {a.planted_dir}")
        benches[name] = [pd.read_csv(f, sep="\t").set_index("Run_accession") for f in files]
    cls_rows, cls_pairs, flag_rows, flag_pairs = [], [], [], []
    for task in TASKS:
        src = a.feature_from if (task == "feature" and a.feature_from is not None) else a.diana
        models = {"DIANA": diana_probs(src, task)}
        for b in BASELINES:
            try:
                models[b] = baseline_probs(a.baselines, task, b)
            except (FileNotFoundError, KeyError):
                continue
        if "LogisticRegression_Bal" in models:
            models["DIANA+LogReg average"] = average(models["DIANA"], models["LogisticRegression_Bal"])
        eligible = load_eligible_classes(a.eligibility, task)
        # LinearSVM_Bal: per-run stored predictions, classification only
        svm = pd.read_csv(a.baselines / "heldout_predictions.tsv", sep="\t")
        svm = svm[(svm.task == task) & (svm.model == "LinearSVM_Bal")].set_index("Run_accession")
        # ---- classification, f1_macro_eligible, paired over projects
        labelled = meta.index[meta[task].notna()]
        projects = np.array(sorted(meta.loc[labelled, "archive_project"].astype(str).unique()))
        rng = rng_for(task, "f1"); draws = [rng.integers(0, len(projects), size=len(projects)) for _ in range(N_BOOT_F1)]
        boots = {}
        cls_models = {name: ("probs", P) for name, P in models.items()}
        if len(svm):
            cls_models["LinearSVM_Bal"] = ("preds", svm)
        for name, (kind, P) in cls_models.items():
            runs = P.index.intersection(labelled)
            y = meta.loc[runs, task].astype(str).to_numpy()
            pred = (P.loc[runs].idxmax(axis=1) if kind == "probs" else P.loc[runs, "y_pred"]).astype(str).to_numpy()
            codes = pd.Categorical(meta.loc[runs, "archive_project"].astype(str), categories=projects).codes
            point = classification_metrics(y, pred, eligible)
            bs = []
            for draw in draws:
                w = np.bincount(draw, minlength=len(projects))[codes]; idx = np.repeat(np.arange(len(y)), w)
                bs.append(classification_metrics(y[idx], pred[idx], eligible)["f1_macro_eligible"] if len(idx) else np.nan)
            boots[name] = np.asarray(bs, float); lo, hi = ci(boots[name])
            cls_rows.append({"task": task, "model": name, "n_runs": len(runs), "n_projects": meta.loc[runs, "archive_project"].nunique(),
                             "f1_macro_eligible": point["f1_macro_eligible"], "ci_low": lo, "ci_high": hi,
                             "n_classes_eligible": point["n_classes_eligible"], "balanced_accuracy": point["balanced_accuracy"]})
        for x, yn in [("DIANA", "LogisticRegression_Bal"), ("DIANA+LogReg average", "DIANA"), ("DIANA+LogReg average", "LogisticRegression_Bal")]:
            if x in boots and yn in boots:
                d = boots[x] - boots[yn]; lo, hi = ci(d)
                obs = next(r["f1_macro_eligible"] for r in cls_rows if r["task"] == task and r["model"] == x) - next(r["f1_macro_eligible"] for r in cls_rows if r["task"] == task and r["model"] == yn)
                cls_pairs.append({"task": task, "a": x, "b": yn, "delta_f1": obs, "ci_low": lo, "ci_high": hi, "frac_boot_positive": float(np.nanmean(d > 0)),
                                  "verdict": "tie" if lo <= 0 <= hi else (x if obs > 0 else yn)})
        # ---- flagging, both benchmarks, averaged over the seeded plantings (as on the dev folds)
        for bench, plants in benches.items():
            tables = {}
            for name, P in models.items():
                per = []
                for pl in plants:
                    flag_col = f"{task}_planted"
                    had = meta.loc[pl.index, task].notna().to_numpy()
                    pl_t = pl[pl[task].notna() & pl[flag_col].notna() & had]
                    sc = F.flag_scores(P, pl_t[task]).dropna()
                    per.append(pd.DataFrame({"score": sc.to_numpy(float), "planted": pl_t.loc[sc.index, flag_col].astype(bool).to_numpy(),
                                             "project": meta.loc[sc.index, "archive_project"].astype(str).to_numpy()}, index=sc.index))
                tables[name] = per
            fprojects = projects
            rng = rng_for(task, "flag", bench); fdraws = [rng.integers(0, len(fprojects), size=len(fprojects)) for _ in range(N_BOOT_FLAG)]
            fboot = {}
            for name, per in tables.items():
                point = {k: [] for k in ("r5", "r10", "r20", "cs", "nb", "sat")}
                for t in per:
                    sc, pf = t.score.to_numpy(), t.planted.to_numpy()
                    point["r5"].append(F.recall_at_budget(sc, pf, 0.05)[0]); point["r10"].append(F.recall_at_budget(sc, pf, 0.10)[0])
                    point["r20"].append(F.recall_at_budget(sc, pf, 0.20)[0])
                    cs_, nb_ = F.choosing_score(sc, pf); point["cs"].append(cs_); point["nb"].append(nb_); point["sat"].append(F.saturated_share(sc, pf))
                codes = [pd.Categorical(t.project, categories=fprojects).codes for t in per]
                b5, bcs = [], []
                for draw in fdraws:
                    counts = np.bincount(draw, minlength=len(fprojects))
                    r5s, css = [], []
                    for t, c in zip(per, codes):
                        idx = np.repeat(np.arange(len(t)), counts[c])
                        if len(idx) == 0:
                            continue
                        sc, pf = t.score.to_numpy()[idx], t.planted.to_numpy()[idx]
                        r5s.append(F.recall_at_budget(sc, pf, 0.05)[0]); css.append(F.choosing_score(sc, pf)[0])
                    b5.append(np.nanmean(r5s) if r5s else np.nan)
                    bcs.append(np.nanmean(css) if css and np.isfinite(css).any() else np.nan)
                fboot[name] = (np.asarray(b5, float), np.asarray(bcs, float)); lo, hi = ci(fboot[name][0]); clo, chi = ci(fboot[name][1])
                sat = float(np.nanmean(point["sat"]))
                flag_rows.append({"bench": bench, "task": task, "model": name, "n_plantings": len(per),
                                  "runs_scored_mean": float(np.mean([len(t) for t in per])), "planted_mean": float(np.mean([t.planted.sum() for t in per])),
                                  "recall_at_5pct": float(np.nanmean(point["r5"])), "ci_low_5pct": lo, "ci_high_5pct": hi,
                                  "recall_at_10pct": float(np.nanmean(point["r10"])), "recall_at_20pct": float(np.nanmean(point["r20"])),
                                  "choosing_score": float(np.nanmean(point["cs"])) if np.isfinite(point["cs"]).any() else np.nan,
                                  "choosing_ci_low": clo, "choosing_ci_high": chi,
                                  "budgets_attainable": float(np.min(point["nb"])), "saturated_share": sat, "cant_check_5pct": sat > 0.05})
            for x, yn in [("DIANA", "LogisticRegression_Bal"), ("DIANA+LogReg average", "DIANA"), ("DIANA+LogReg average", "LogisticRegression_Bal")]:
                if x in fboot and yn in fboot:
                    d = fboot[x][1] - fboot[yn][1]; d5 = fboot[x][0] - fboot[yn][0]; lo, hi = ci(d); lo5, hi5 = ci(d5)
                    rx = next(r for r in flag_rows if r.get("bench") == bench and r["task"] == task and r["model"] == x)
                    ry = next(r for r in flag_rows if r.get("bench") == bench and r["task"] == task and r["model"] == yn)
                    obs = rx["choosing_score"] - ry["choosing_score"]
                    flag_pairs.append({"bench": bench, "task": task, "a": x, "b": yn, "delta_choosing": obs, "ci_low": lo, "ci_high": hi,
                                       "verdict": "can't check" if not np.isfinite(obs) or not np.isfinite(lo) else "tie" if lo <= 0 <= hi else (x if obs > 0 else yn),
                                       "delta_recall_at_5pct": rx["recall_at_5pct"] - ry["recall_at_5pct"], "ci_low_5pct": lo5, "ci_high_5pct": hi5})
    C, CP, FL, FP = map(pd.DataFrame, (cls_rows, cls_pairs, flag_rows, flag_pairs))
    C.to_csv(a.out / "classification.tsv", sep="\t", index=False); CP.to_csv(a.out / "classification_paired.tsv", sep="\t", index=False)
    FL.to_csv(a.out / "flagging.tsv", sep="\t", index=False); FP.to_csv(a.out / "flagging_paired.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("HELD-OUT READ 11, classification (f1_macro_eligible, BioProject intervals):"); print(C.round(3).to_string(index=False))
    print("\npaired differences:"); print(CP.round(3).to_string(index=False))
    print("\nflagging on the held-out plantings:"); print(FL.round(3).to_string(index=False))
    print("\npaired differences (choosing score, recall at 5 %):"); print(FP.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
