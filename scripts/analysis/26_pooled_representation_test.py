#!/usr/bin/env python3
"""Compare feature representations on pooled out-of-fold predictions, paired over BioProjects.

Why this replaces the per-fold comparison
-----------------------------------------
Each representation was scored on 5 BioProject-grouped dev folds, and the earlier
comparison collapsed those to 5 fold-means and put an interval on n = 5. That discards
almost all the information: every one of the ~2,716 training runs has an out-of-fold
prediction from a model that never saw its BioProject, so the predictions pool into one
clean set and the resampling unit can be the **BioProject**, of which train has 68. That
is the same unit the held-out numbers use, and `CLAUDE.md` requires it.

It also fixes the metric. `f1_macro_eligible` restricts to classes present in >= 2
BioProjects. That rule was written for held-out's 34 projects; a single dev fold has
about 13, so between 1 and 9 classes qualified and three task-folds had exactly one,
making "macro" a single class's F1. Pooling restores the full class set and the
eligibility rule is applied once, over all 68 training projects.

Pairing is the point: both representations predict the same runs, so between-project
difficulty cancels and the interval is on the difference, not on two overlapping
absolutes.

    ./env/bin/python scripts/analysis/26_pooled_representation_test.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/representation_pooled"

TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = {"fraction": ("results/calibration_v9", ""),
        "abundance": ("results/abundance_test_v9", ""),
        "presence_absence": ("results/pa_test_v9", ""),
        # G5/G6: reduced spaces, one subdirectory prefix per arm
        "frac_pca192": ("results/reduction_test_v9", "frac_pca192_"),
        "pa_pca192": ("results/reduction_test_v9", "pa_pca192_"),
        "frac_pca191nd": ("results/reduction_test_v9", "frac_pca191nd_"),
        "pa_pca191nd": ("results/reduction_test_v9", "pa_pca191nd_"),
        # S1: fractions with 136 canonical 4-mer composition columns appended
        "frac_plus_kmer4": ("results/kmer_test_v9", ""),
        # S2: fractions with 6,733 cluster-sum columns appended
        "frac_plus_clusters": ("results/clust_test_v9", ""),
        # S4: the input layer's per-unitig weights computed from bases by one shared CNN
        "seq_encoder": ("results/seqenc_test_v9", ""),
        # S4 control: identical, with each unitig's bases permuted within itself, so base
        # composition and length survive and only ORDER is destroyed. A gain that appears
        # in both arms is capacity or regularisation, not sequence.
        "seq_encoder_shuffled": ("results/seqenc_shuf_v9", "")}
N_BOOT = 2000
SEED = 42
# Few-shot is what the S arms exist to fix (Table 2 reads 0.000 on `material` few-shot),
# so a whole-task tie must not hide a regime-level effect. Support bands are the
# ImageNet-LT / OLTR split points on TRAINING runs per class.
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, float("inf"))]
# G7 asks a different question from the rest: not "beats the fraction" but "does the
# library-size component carry usable signal". Each pair is (with depth, without depth).
DEPTH_PAIRS = [("frac_pca192", "frac_pca191nd"), ("pa_pca192", "pa_pca191nd")]
MIN_PROJECTS = 2          # the eligibility rule, applied once over pooled training data


def pooled(base: str, task: str, prefix: str = "") -> pd.DataFrame:
    """Every out-of-fold prediction for one task, concatenated across the 5 folds."""
    frames = []
    for f in range(5):
        p = ROOT / base / f"oof_{prefix}{task}_fold{f}/test_predictions.tsv"
        if not p.exists():
            continue
        d = pd.read_csv(p, sep="\t", usecols=["Run_accession", f"{task}_pred", f"{task}_true"])
        d["fold"] = f
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no predictions under {base} for {task}")
    d = pd.concat(frames, ignore_index=True)
    dup = d.Run_accession.duplicated().sum()
    if dup:
        logger.warning("%s %s: %d runs appear in more than one fold", base, task, dup)
    return d


def f1_elig(y, p, eligible) -> float:
    labs = sorted(c for c in set(y) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    proj = meta.set_index("Run_accession")["archive_project"].to_dict()
    rng = np.random.default_rng(SEED)
    rows, summary, depth_rows = [], [], []

    for task in TASKS:
        arms = {}
        for name, (base, prefix) in ARMS.items():
            # An arm that is absent or half-finished must be skipped, not tolerated. The
            # scored set below is the INTERSECTION over arms, so an arm holding 3 of 5 folds
            # would quietly shrink every other arm's run count and silently change numbers
            # that are already in the manuscript. Crashing is no better: it would block the
            # finished arms from being scored at all while a new arm is still running.
            try:
                d = pooled(base, task, prefix)
            except FileNotFoundError:
                logger.warning("%s: arm %r has no predictions, skipped", task, name)
                continue
            if d.fold.nunique() < 5:
                logger.warning("%s: arm %r has %d of 5 folds, skipped as incomplete",
                               task, name, d.fold.nunique())
                continue
            d = d[d[f"{task}_true"].notna()]
            d["project"] = d.Run_accession.map(proj)
            arms[name] = d.set_index("Run_accession")

        if "fraction" not in arms:
            raise SystemExit(f"{task}: the fraction baseline is missing; nothing to compare against")

        common = set.intersection(*(set(d.index) for d in arms.values()))
        logger.info("%s: %d runs scored by all %d arms", task, len(common), len(arms))
        idx = sorted(common)
        base_d = arms["fraction"].loc[idx]
        y = base_d[f"{task}_true"].astype(str).to_numpy()
        g = base_d["project"].to_numpy()

        # eligibility applied ONCE over the pooled training projects
        per_class = pd.DataFrame({"c": y, "g": g}).groupby("c").g.nunique()
        eligible = set(per_class[per_class >= MIN_PROJECTS].index)
        summary.append({"task": task, "n_runs": len(idx), "n_projects": len(set(g)),
                        "n_classes": len(per_class), "n_eligible": len(eligible)})
        logger.info("   %d projects, %d classes, %d eligible (>=%d projects)",
                    len(set(g)), len(per_class), len(eligible), MIN_PROJECTS)

        preds = {n: arms[n].loc[idx][f"{task}_pred"].astype(str).to_numpy() for n in arms}
        uniq = np.unique(g)
        idx_by = {p: np.where(g == p)[0] for p in uniq}

        for name in [a for a in ARMS if a != "fraction"]:
            obs = f1_elig(y, preds[name], eligible) - f1_elig(y, preds["fraction"], eligible)
            deltas = []
            for _ in range(N_BOOT):
                drawn = rng.choice(uniq, size=len(uniq), replace=True)
                sel = np.concatenate([idx_by[p] for p in drawn])
                deltas.append(f1_elig(y[sel], preds[name][sel], eligible)
                              - f1_elig(y[sel], preds["fraction"][sel], eligible))
            dd = np.asarray(deltas, float)
            dd = dd[np.isfinite(dd)]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append({"task": task, "arm": name,
                         "f1_fraction": f1_elig(y, preds["fraction"], eligible),
                         "f1_arm": f1_elig(y, preds[name], eligible),
                         "delta": obs, "ci_low": lo, "ci_high": hi,
                         "frac_boot_positive": float((dd > 0).mean()),
                         "n_projects": len(uniq), "n_runs": len(idx),
                         "verdict": "tie" if lo <= 0 <= hi else
                                    ("BETTER" if obs > 0 else "WORSE")})

        # G7: paired test between the with-depth and without-depth arms directly
        for a, b in DEPTH_PAIRS:
            obs = f1_elig(y, preds[a], eligible) - f1_elig(y, preds[b], eligible)
            dd = []
            for _ in range(N_BOOT):
                drawn = rng.choice(uniq, size=len(uniq), replace=True)
                sel = np.concatenate([idx_by[q] for q in drawn])
                dd.append(f1_elig(y[sel], preds[a][sel], eligible)
                          - f1_elig(y[sel], preds[b][sel], eligible))
            dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            depth_rows.append({"task": task, "comparison": f"{a} - {b}",
                               "f1_with_depth": f1_elig(y, preds[a], eligible),
                               "f1_without_depth": f1_elig(y, preds[b], eligible),
                               "delta": obs, "ci_low": lo, "ci_high": hi,
                               "frac_boot_positive": float((dd > 0).mean()),
                               "verdict": "tie" if lo <= 0 <= hi else
                                          ("depth HELPS" if obs > 0 else "depth HURTS")})

    res = pd.DataFrame(rows)
    pd.DataFrame(depth_rows).to_csv(OUT / "depth_component.tsv", sep="\t", index=False)
    print("\nG7 — keeping vs dropping the library-size component, paired over BioProjects:")
    print(pd.DataFrame(depth_rows).round(4).to_string(index=False))
    pd.DataFrame(summary).to_csv(OUT / "pooled_summary.tsv", sep="\t", index=False)
    res.to_csv(OUT / "paired_vs_fraction.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("\npooled out-of-fold, eligibility applied once over all training projects:")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\npaired over BioProjects, 2,000 resamples, vs the fraction:")
    print(res.round(4).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
