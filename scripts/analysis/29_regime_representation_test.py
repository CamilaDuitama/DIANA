#!/usr/bin/env python3
"""Every representation arm against the fraction, split by training-support regime.

`26_pooled_representation_test.py` reports one number per task. That is not enough for
the S arms: they exist to fix the bottom of Table 2, where `material` few-shot reads
**0.000** across all eight models, and a whole-task tie can hide a real few-shot effect
in either direction. The S-plan discipline in PROJECT.md requires the split.

Same machinery as the pooled test: out-of-fold predictions concatenated across the 5
dev folds, the eligibility rule (class in >= 2 BioProjects) applied **once** over all
training projects, and the paired difference bootstrapped over BioProjects. The only
change is that macro-F1 is averaged over the eligible classes whose TRAINING support
falls in one band, while still scoring every row -- filtering rows to the band would
discard the false positives rare classes attract from elsewhere, which is most of what
makes them hard.

Bands are the ImageNet-LT / OLTR split points: few-shot < 20 training runs, medium-shot
20-100, many-shot > 100.

    ./env/bin/python scripts/analysis/29_regime_representation_test.py
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
        "frac_plus_kmer4": ("results/kmer_test_v9", ""),
        "frac_plus_clusters": ("results/clust_test_v9", ""),
        # S4: the input layer's per-unitig weights computed from bases by one shared CNN
        "seq_encoder": ("results/seqenc_test_v9", ""),
        # S4 control: identical, with each unitig's bases permuted within itself, so base
        # composition and length survive and only ORDER is destroyed. A gain that appears
        # in both arms is capacity or regularisation, not sequence.
        "seq_encoder_shuffled": ("results/seqenc_shuf_v9", "")}
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT = 2000
SEED = 42


def pooled(base: str, task: str, prefix: str) -> pd.DataFrame | None:
    fr = []
    for f in range(5):
        p = ROOT / base / f"oof_{prefix}{task}_fold{f}/test_predictions.tsv"
        if p.exists():
            fr.append(pd.read_csv(p, sep="\t",
                                  usecols=["Run_accession", f"{task}_pred", f"{task}_true"]))
    return pd.concat(fr, ignore_index=True) if fr else None


def f1_band(y, p, band) -> float:
    labs = sorted(c for c in set(y) if c in band)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    proj = meta.set_index("Run_accession")["archive_project"].to_dict()
    rng = np.random.default_rng(SEED)
    rows = []

    for task in TASKS:
        sup = meta[task].value_counts()
        arms = {}
        for name, (base, pre) in ARMS.items():
            d = pooled(base, task, pre)
            if d is None:
                logger.warning("%s: missing %s", task, name); continue
            d = d[d[f"{task}_true"].notna()]
            arms[name] = d.set_index("Run_accession")
        if "fraction" not in arms or len(arms) < 2:
            continue
        idx = sorted(set.intersection(*(set(v.index) for v in arms.values())))
        y = arms["fraction"].loc[idx, f"{task}_true"].astype(str).to_numpy()
        g = np.array([proj.get(i) for i in idx])
        per = pd.DataFrame({"c": y, "g": g}).groupby("c").g.nunique()
        eligible = set(per[per >= 2].index)
        uniq = np.unique(g)
        idx_by = {p: np.where(g == p)[0] for p in uniq}
        preds = {n: arms[n].loc[idx, f"{task}_pred"].astype(str).to_numpy() for n in arms}

        for nm, lo, hi in REGIMES:
            band = {c for c in eligible if lo < sup.get(c, 0) <= hi}
            present = band & set(y)
            if not present:
                continue
            for name in [a for a in arms if a != "fraction"]:
                obs = f1_band(y, preds[name], band) - f1_band(y, preds["fraction"], band)
                dd = []
                for _ in range(N_BOOT):
                    drawn = rng.choice(uniq, size=len(uniq), replace=True)
                    s = np.concatenate([idx_by[q] for q in drawn])
                    dd.append(f1_band(y[s], preds[name][s], band)
                              - f1_band(y[s], preds["fraction"][s], band))
                dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
                if dd.size < 100:
                    continue
                lo_, hi_ = np.percentile(dd, [2.5, 97.5])
                rows.append({"task": task, "regime": nm, "arm": name,
                             "n_classes": len(present),
                             "f1_fraction": f1_band(y, preds["fraction"], band),
                             "f1_arm": f1_band(y, preds[name], band),
                             "delta": obs, "ci_low": lo_, "ci_high": hi_,
                             "frac_boot_positive": float((dd > 0).mean()),
                             "verdict": "tie" if lo_ <= 0 <= hi_ else
                                        ("BETTER" if obs > 0 else "WORSE")})
            logger.info("  %s / %s: %d eligible classes present", task, nm, len(present))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "by_regime.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print("\nper regime, paired over BioProjects, vs the fraction:")
    print(res.round(4).to_string(index=False))
    print("\nverdicts:", res.verdict.value_counts().to_dict())
    hits = res[res.verdict != "tie"]
    if not hits.empty:
        print("\nnon-ties (corroboration rule: the same arm and regime must clear on >1 task):")
        print(hits[["task", "regime", "arm", "delta", "ci_low", "ci_high", "verdict"]]
              .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
