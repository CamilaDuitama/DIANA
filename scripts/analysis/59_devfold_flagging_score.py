#!/usr/bin/env python3
"""X9.0b to X9.0d: the flagging score on the grouped dev folds.

For every model with out-of-fold probabilities (the frozen reference, as the 5-seed
average of results/reference_v9/<task>/seed<s>/fold<k>/probs_last.npz, and each baseline
in results/devfold_probs_v9/) and every planting in results/planted_mislabels_dev_v9/
(single-label plantings on the training metadata, one file per seed):

  flag score      = 1 - P(label on file), the planted label where planted
  threshold       = the value that flags 5 % of the runs whose label is correct
  recall@5%       = share of planted runs above that threshold
  ROC-AUC         = planted vs clean, threshold-free (reported, not the decision score)
  false flags     = clean runs above the threshold, by true class (X9.0d)

Scores are averaged over the plantings. Intervals are bootstraps over BioProjects: a
resample draws projects, keeps all their runs, recomputes threshold and recall inside
every planting and averages; 1,000 resamples. The reference minus each baseline is the
paired difference on the same resamples. Runs whose label on file is a class the model
has no probability for are dropped for that model and counted.

Outputs, results/devfold_flagging_v9/: flagging_scores.tsv (per model and task),
flagging_paired.tsv (reference minus baseline), false_flags_by_class.tsv, and a per-task
summary printed.

    ./env/bin/python scripts/analysis/59_devfold_flagging_score.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
PLANT_DIR, REF_DIR, BASE_DIR = ROOT / "results/planted_mislabels_dev_v9", ROOT / "results/reference_v9", ROOT / "results/devfold_probs_v9"
OUT = ROOT / "results/devfold_flagging_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
BASELINES = ["LogisticRegression_Bal", "RandomForest_Bal", "kNN"]
SEEDS = [42, 1, 2, 3, 4]
BUDGET, N_BOOT, SEED = 0.05, 1000, 42
BUDGETS = (0.05, 0.10, 0.20)   # the choosing score averages the catch rate over these; 5 % is the headline


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def reference_probs(task: str, base: Path = None) -> pd.DataFrame:
    """Run x class probabilities of a 5-seed ensemble in the reference layout, pooled over folds."""
    base = base or REF_DIR
    parts = []
    for k in range(5):
        dirs = [base / task / f"seed{s}" / f"fold{k}" for s in SEEDS]
        runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
        classes = json.load(open(dirs[0] / "label_classes.json"))[task]
        P = np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0)
        parts.append(pd.DataFrame(P, index=runs, columns=classes))
    return pd.concat(parts)


def baseline_probs(model: str, task: str, base: Path = None) -> pd.DataFrame:
    d = pd.read_csv((base or BASE_DIR) / f"probs_{model}_{task}.tsv", sep="\t").set_index("Run_accession")
    cols = [c for c in d.columns if c.startswith("p_")]
    return d[cols].rename(columns={c: c[2:] for c in cols})


def flag_scores(P: pd.DataFrame, stated: pd.Series) -> pd.Series:
    """1 - P(stated label); NaN where the model has no column for that label."""
    common = P.index.intersection(stated.index)
    st = stated.loc[common].astype(str)
    out = pd.Series(np.nan, index=common)
    for c in st.unique():
        if c in P.columns:
            m = st == c
            out[m[m].index] = 1.0 - P.loc[m[m].index, c].to_numpy()
    return out


def recall_at_budget(score: np.ndarray, planted: np.ndarray, budget: float = BUDGET) -> tuple[float, float]:
    """The held-out definition (15_/19_): the last ROC operating point whose false-flag rate is
    at most the budget. Models whose flag scores tie at the top (kNN with k = 1, saturated
    probabilities) cannot operate at a 5 % budget and read 0, exactly as on held-out."""
    if planted.sum() == 0 or (~planted).sum() == 0:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(planted, score)
    i = max(int(np.searchsorted(fpr, budget, side="right") - 1), 0)
    return float(tpr[i]), float(thr[i])


def saturated_share(score: np.ndarray, planted: np.ndarray) -> float:
    """Share of correctly labelled runs at the maximum flag score. If it exceeds a budget, that
    budget cannot be operated at: the cell is "can't check" and is left out of the choosing score."""
    clean = score[~planted]
    return float((clean >= np.nanmax(score) - 1e-9).mean()) if len(clean) else np.nan


def choosing_score(score: np.ndarray, planted: np.ndarray) -> tuple[float, int]:
    """Mean catch rate over the attainable budgets in BUDGETS; (nan, 0) if none is attainable."""
    sat = saturated_share(score, planted)
    vals = [recall_at_budget(score, planted, b)[0] for b in BUDGETS if sat <= b]
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def pair_row(task: str, a: str, b: str, summ: dict, boot: dict, boot_cs: dict) -> dict:
    """a minus b on the choosing score (and recall at 5 %), on the shared project draws."""
    d = np.asarray(boot_cs[a], float) - np.asarray(boot_cs[b], float); d = d[np.isfinite(d)]
    obs = summ[a]["choosing_score"] - summ[b]["choosing_score"]
    d5 = np.asarray(boot[a], float) - np.asarray(boot[b], float); d5 = d5[np.isfinite(d5)]
    return {"task": task, "a": a, "b": b, "baseline": b,
            "reference_choosing": summ[a]["choosing_score"], "baseline_choosing": summ[b]["choosing_score"],
            "delta_choosing": obs, "ci_low": np.percentile(d, 2.5) if len(d) else np.nan,
            "ci_high": np.percentile(d, 97.5) if len(d) else np.nan,
            "frac_boot_positive": float((d > 0).mean()) if len(d) else np.nan,
            "verdict": ("can't check" if not len(d) or np.isnan(obs) else
                        "tie" if np.percentile(d, 2.5) <= 0 <= np.percentile(d, 97.5) else (a if obs > 0 else b)),
            "delta_recall_at_5pct": summ[a]["recall_at_5pct"] - summ[b]["recall_at_5pct"],
            "ci_low_5pct": np.percentile(d5, 2.5) if len(d5) else np.nan, "ci_high_5pct": np.percentile(d5, 97.5) if len(d5) else np.nan}


def main() -> int:
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pattern", default="planted_train_mixed_r0.1_s*.tsv", help="planting files in results/planted_mislabels_dev_v9")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--extra", action="append", default=[],
                    help="candidate ensembles in the reference layout, as name=dir (X9, X10); scored and paired like the baselines")
    ap.add_argument("--extra-probs", action="append", default=[],
                    help="logistic-regression probabilities from another directory, as name=dir (Phase 4 step 4.2: the study-weighted fit)")
    ap.add_argument("--pair", action="append", default=[],
                    help="extra paired rows a:b (a minus b), besides reference minus every other model (e.g. step4_1:LogisticRegression_Bal)")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    plantings = sorted(glob.glob(str(PLANT_DIR / args.pattern)))
    if not plantings:
        raise SystemExit(f"no plantings match {args.pattern}")
    plants = [pd.read_csv(f, sep="\t").set_index("Run_accession") for f in plantings]
    rows, paired_rows, ff_rows = [], [], []
    for task in TASKS:
        models = {"reference": reference_probs(task)}
        for e in args.extra:
            name, d = e.split("=", 1)
            models[name] = reference_probs(task, Path(d))
        for b in BASELINES:
            try:
                models[b] = baseline_probs(b, task)
            except FileNotFoundError:
                continue
        for e in args.extra_probs:
            name, d = e.split("=", 1)
            models[name] = baseline_probs("LogisticRegression_Bal", task, Path(d))
        # per model and planting: flag score, planted flag, project, true label
        tables = {}
        for name, P in models.items():
            per = []
            for pi, pl in enumerate(plants):
                flag_col = f"{task}_planted"
                # a run can only be clean or planted if it had a label to begin with; a run that
                # received a label through a record swap but had none is neither, and is excluded
                had_label = meta.loc[pl.index, task].notna().to_numpy()
                pl_t = pl[pl[task].notna() & pl[flag_col].notna() & had_label]
                s = flag_scores(P, pl_t[task])
                s = s.dropna()
                per.append(pd.DataFrame({"score": s, "planted": pl_t.loc[s.index, flag_col].astype(bool),
                                         "project": meta.loc[s.index, "archive_project"].astype(str),
                                         "true": meta.loc[s.index, task].astype(str),
                                         "material": meta.loc[s.index, "material"].astype(str), "planting": pi}))
            tables[name] = per
        # point estimates averaged over plantings
        def summarise(per, exclude_tooth=False):
            out = {f"recall_at_{int(b*100)}pct": [] for b in BUDGETS}
            out.update({"choosing_score": [], "budgets_attainable": [], "saturated_share": [], "roc_auc": []})
            for t in per:
                if exclude_tooth:
                    t = t[t.material != "tooth"]
                sc, pl_ = t.score.to_numpy(), t.planted.to_numpy()
                for b in BUDGETS:
                    out[f"recall_at_{int(b*100)}pct"].append(recall_at_budget(sc, pl_, b)[0])
                cs, nb = choosing_score(sc, pl_)
                out["choosing_score"].append(cs); out["budgets_attainable"].append(nb); out["saturated_share"].append(saturated_share(sc, pl_))
                out["roc_auc"].append(roc_auc_score(pl_, sc) if pl_.any() and (~pl_).any() else np.nan)
            return {k: float(np.nanmean(v)) if k != "budgets_attainable" else float(np.min(v)) for k, v in out.items()}
        # bootstrap over projects, shared resamples across models for the paired difference
        projects = np.unique(np.concatenate([t.project.unique() for per in tables.values() for t in per]))
        rng = rng_for(task)
        boot = {name: [] for name in tables}; boot_cs = {name: [] for name in tables}
        for _ in range(N_BOOT):
            draw = rng.choice(projects, size=len(projects), replace=True)
            counts = pd.Series(draw).value_counts()
            for name, per in tables.items():
                recs, css = [], []
                for t in per:
                    w = t.project.map(counts).fillna(0).astype(int).to_numpy()
                    idx = np.repeat(np.arange(len(t)), w)
                    if len(idx) == 0:
                        continue
                    sc, pl_ = t.score.to_numpy()[idx], t.planted.to_numpy()[idx]
                    recs.append(recall_at_budget(sc, pl_)[0]); css.append(choosing_score(sc, pl_)[0])
                boot[name].append(np.nanmean(recs) if recs else np.nan)
                boot_cs[name].append(np.nanmean(css) if css and np.isfinite(css).any() else np.nan)
        summ = {name: summarise(per) for name, per in tables.items()}
        summ_nt = {name: summarise(per, exclude_tooth=True) for name, per in tables.items()} if task == "community_type" else {}
        for name, per in tables.items():
            sm = summ[name]
            b = np.asarray(boot[name], float); b = b[np.isfinite(b)]
            bc = np.asarray(boot_cs[name], float); bc = bc[np.isfinite(bc)]
            n_planted = float(np.mean([t.planted.sum() for t in per])); n_scored = float(np.mean([len(t) for t in per]))
            dropped = float(np.mean([len(pl[pl[task].notna()]) for pl in plants]) - n_scored)
            cant = sm["saturated_share"] > BUDGET
            rows.append({"task": task, "model": name, "n_plantings": len(per), "runs_scored_mean": n_scored, "planted_mean": n_planted,
                         "runs_dropped_no_probability_mean": dropped,
                         "recall_at_5pct": sm["recall_at_5pct"], "ci_low": np.percentile(b, 2.5) if len(b) else np.nan,
                         "ci_high": np.percentile(b, 97.5) if len(b) else np.nan,
                         "recall_at_10pct": sm["recall_at_10pct"], "recall_at_20pct": sm["recall_at_20pct"],
                         "choosing_score": sm["choosing_score"], "choosing_ci_low": np.percentile(bc, 2.5) if len(bc) else np.nan,
                         "choosing_ci_high": np.percentile(bc, 97.5) if len(bc) else np.nan,
                         "budgets_attainable": sm["budgets_attainable"], "saturated_share": sm["saturated_share"],
                         "cant_check_5pct": cant, "roc_auc": sm["roc_auc"],
                         "tooth_excluded_recall_at_5pct": summ_nt[name]["recall_at_5pct"] if summ_nt else np.nan,
                         "tooth_excluded_choosing_score": summ_nt[name]["choosing_score"] if summ_nt else np.nan})
            if name != "reference":
                paired_rows.append(pair_row(task, "reference", name, summ, boot, boot_cs))
        for pr_ in args.pair:
            a_, b_ = pr_.split(":", 1)
            if a_ in tables and b_ in tables:
                paired_rows.append(pair_row(task, a_, b_, summ, boot, boot_cs))
        for name, per in tables.items():
            # false flags by true class, clean runs, averaged over plantings
            ff = []
            for t in per:
                _, thr = recall_at_budget(t.score.to_numpy(), t.planted.to_numpy())
                clean = t[~t.planted]
                ff.append(clean.assign(flag=clean.score >= thr).groupby("true").flag.agg(["mean", "size"]))
            ffm = pd.concat(ff).groupby(level=0).agg({"mean": "mean", "size": "mean"}).reset_index()
            for _, r in ffm.iterrows():
                ff_rows.append({"task": task, "model": name, "true_class": r["true"], "clean_runs_mean": r["size"], "false_flag_rate": r["mean"]})
    res, pr, ff = pd.DataFrame(rows), pd.DataFrame(paired_rows), pd.DataFrame(ff_rows)
    res.to_csv(OUT / "flagging_scores.tsv", sep="\t", index=False)
    pr.to_csv(OUT / "flagging_paired.tsv", sep="\t", index=False)
    ff.to_csv(OUT / "false_flags_by_class.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("headline recall at 5 % (CI over BioProjects), recall at 10 % and 20 %, the choosing score (mean over attainable budgets), can't-check:")
    print(res[["task", "model", "planted_mean", "recall_at_5pct", "ci_low", "ci_high", "recall_at_10pct", "recall_at_20pct",
               "choosing_score", "choosing_ci_low", "choosing_ci_high", "budgets_attainable", "saturated_share", "cant_check_5pct", "roc_auc"]].round(3).to_string(index=False))
    print("\ntooth-excluded (community_type only, explanation, never for choosing):")
    print(res[res.task == "community_type"][["model", "recall_at_5pct", "tooth_excluded_recall_at_5pct", "choosing_score", "tooth_excluded_choosing_score"]].round(3).to_string(index=False))
    print("\nreference minus baseline on the choosing score, paired (and the 5 % difference):")
    print(pr.round(3).to_string(index=False))
    print("\nfalse flags by true class (clean runs, rate above 0.10 shown):")
    print(ff[(ff.false_flag_rate > 0.10) & (ff.clean_runs_mean >= 10)].round(3).sort_values(["task", "model", "false_flag_rate"], ascending=[True, True, False]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
