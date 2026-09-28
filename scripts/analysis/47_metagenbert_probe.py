#!/usr/bin/env python3
"""S8.7: the MetagenBERT aggregation on the S7 vectors, probed on the grouped dev folds.

The S7 tables (data/sequences_v9/offlist/): up to 2,000 off-list unitigs per sample,
each a 768-d DNABERT-2 vector, real (emb_shard*.npy) and within-sequence scrambled
(emb.shuffled_shard*.npy), with shard*.npz giving the sample of every row. MetagenBERT
represents a sample as the proportions of its sequences falling in each of C clusters of
a global k-means; here the centres are fitted, per dev fold, on rows of the training-fold
samples only (a 1 M-row subsample), and every training run's rows are then assigned.

Probe, per task: balanced logistic regression on (a) the C proportions alone, (b) the
raw fractions alone at the C tuned on the dev folds, (c) both; out-of-fold, pooled,
f1_macro_eligible, paired over projects: (a) real vs scrambled, (c) vs (b). Scaffolding.

    ./env/bin/python scripts/analysis/47_metagenbert_probe.py --clusters 128 512
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data/sequences_v9/offlist"
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/s8a_probe_v9/metagenbert"
TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot", 0, 20), ("medium-shot", 20, 100), ("many-shot", 100, np.inf)]
N_BOOT, SEED, FIT_ROWS = 2000, 42, 1_000_000


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (SEED,) + parts)).encode()).hexdigest()[:8], 16))


def f1_elig(y, p, eligible):
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def paired(y, pa, pb, g, eligible, key):
    uniq = np.unique(g)
    by = {p: np.where(g == p)[0] for p in uniq}
    obs = f1_elig(y, pa, eligible) - f1_elig(y, pb, eligible)
    rng = rng_for(*key)
    dd = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        dd.append(f1_elig(y[sel], pa[sel], eligible) - f1_elig(y[sel], pb[sel], eligible))
    dd = np.asarray(dd, float); dd = dd[np.isfinite(dd)]
    lo, hi = (np.percentile(dd, [2.5, 97.5]) if len(dd) else (np.nan, np.nan))
    return {"f1_a": f1_elig(y, pa, eligible), "f1_b": f1_elig(y, pb, eligible), "delta": obs, "ci_low": lo,
            "ci_high": hi, "frac_boot_positive": float((dd > 0).mean()) if len(dd) else np.nan,
            "n_projects": len(uniq), "verdict": "tie" if lo <= 0 <= hi else ("A" if obs > 0 else "B")}


def load_arm(tag: str, wanted: set[str]):
    """Rows of every wanted sample, concatenated, with the sample of each row."""
    blocks, owner = [], []
    for sf in sorted(glob.glob(str(OFF / "shard*.npz"))):
        idx = Path(sf).stem.replace("shard", "")
        z = np.load(sf, allow_pickle=True)
        E = np.load(OFF / f"emb{tag}_shard{idx}.npy", mmap_mode="r")
        at = 0
        for s, c in zip(z["samples"], z["counts"]):
            if c and str(s) in wanted:
                blocks.append(np.asarray(E[at:at + c], dtype=np.float32)); owner.append(np.full(c, str(s)))
            at += c
    return np.vstack(blocks), np.concatenate(owner)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clusters", type=int, nargs="+", default=[128, 512])
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = [l.strip() for l in open(SPLITS / "train_accessions.txt") if l.strip()]
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession")["fold"]
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    tuned = pd.read_csv(ROOT / "results/baselines_tuned_v9/best_per_model.tsv", sep="\t")
    tuned_C = {r.task: json.loads(r.params)["C"] for r in tuned[tuned.model == "LogisticRegression_Bal"].itertuples()}
    with np.load(ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz", allow_pickle=False) as z:
        F_ids = list(z["sample_ids"].astype(str)); F = z["frac"].astype(np.float32)
    F = pd.DataFrame(F, index=F_ids)

    profiles = {}   # (tag, C, fold) -> DataFrame samples x C (training runs only)
    for tag in ("", ".shuffled"):
        E, owner = load_arm(tag, set(train))
        logger.info("arm %r: %d rows from %d runs in %.0f s", tag or "real", len(E), len(set(owner)), time.time() - t0)
        fold_of_row = folds.reindex(owner).to_numpy()
        for C in a.clusters:
            for k in range(5):
                fit_rows = np.where(fold_of_row != k)[0]
                rng = np.random.default_rng(SEED + k)
                sub = rng.choice(fit_rows, size=min(FIT_ROWS, len(fit_rows)), replace=False)
                km = MiniBatchKMeans(n_clusters=C, batch_size=8192, n_init=3, random_state=SEED).fit(E[sub])
                lab = np.concatenate([km.predict(E[i:i + 500_000]) for i in range(0, len(E), 500_000)])
                prof = pd.crosstab(owner, lab).reindex(columns=range(C), fill_value=0)
                prof = prof.div(prof.sum(1), axis=0)
                profiles[(tag, C, k)] = prof
                logger.info("%s C=%d fold %d: centres on %d rows, profiles %s (%.0f s)", tag or "real", C, k, len(sub), prof.shape, time.time() - t0)
        del E, owner

    rows = []
    for task in TASKS:
        eligible = set(elig_tbl[(elig_tbl.target == task) & elig_tbl.evaluable]["class"].astype(str))
        counts = meta[task].dropna().astype(str).value_counts()
        y_all = meta[task].reindex(train).astype(object)
        lab = y_all.notna().to_numpy()
        f_all = folds.reindex(train).to_numpy()
        for C in a.clusters:
            preds = {arm: np.full(len(train), None, dtype=object) for arm in ("real", "scrambled", "fractions", "both")}
            for k in range(5):
                tr = (f_all != k) & lab; te = (f_all == k) & lab
                ytr = y_all[tr].astype(str).to_numpy()
                inputs = {}
                for arm, tag in (("real", ""), ("scrambled", ".shuffled")):
                    P = profiles[(tag, C, k)].reindex(train).fillna(0).to_numpy()
                    inputs[arm] = (P[tr], P[te])
                Fr = F.reindex(train).to_numpy()
                inputs["fractions"] = (Fr[tr], Fr[te])
                Pr = profiles[("", C, k)].reindex(train).fillna(0).to_numpy()
                inputs["both"] = (np.hstack([Fr[tr], Pr[tr]]), np.hstack([Fr[te], Pr[te]]))
                for arm, (Xtr, Xte) in inputs.items():
                    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=tuned_C[task] if arm != "real" and arm != "scrambled" else 1.0)
                    clf.fit(Xtr, ytr); preds[arm][te] = clf.predict(Xte)
                logger.info("%s C=%d fold %d probed (%.0f s)", task, C, k, time.time() - t0)
            y = y_all[lab].astype(str).to_numpy(); g = meta.loc[np.array(train)[lab], "archive_project"].astype(str).to_numpy()
            P = {arm: preds[arm][lab].astype(str) for arm in preds}
            for arm in P:
                rows.append({"task": task, "C": C, "comparison": f"f1 {arm}", "regime": "all", "f1_a": f1_elig(y, P[arm], eligible)})
            for name, pa, pb in (("real - scrambled", "real", "scrambled"), ("both - fractions", "both", "fractions"), ("real - fractions", "real", "fractions")):
                rows.append({"task": task, "C": C, "comparison": name, "regime": "all", **paired(y, P[pa], P[pb], g, eligible, (task, C, name, "all"))})
                for rn, lo_, hi_ in REGIMES:
                    cls = {c for c, n in counts.items() if lo_ < n <= hi_} & eligible
                    if cls:
                        rows.append({"task": task, "C": C, "comparison": name, "regime": rn, **paired(y, P[pa], P[pb], g, cls, (task, C, name, rn))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "metagenbert_probe.tsv", sep="\t", index=False)
    pd.set_option("display.width", 240)
    print(res.round(3).to_string(index=False))
    logger.info("done in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
