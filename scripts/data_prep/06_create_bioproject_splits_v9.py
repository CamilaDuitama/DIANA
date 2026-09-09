#!/usr/bin/env python3
"""Create the v9 BioProject-disjoint partition (PROJECT.md G2-G6).

Question answered
-----------------
R3.2 asked for a BioProject-disjoint train/test split. v7's problem was never the
definition -- it was that the split was asserted in a JSON file and never checked.
`data/splits_v7/split_config.json` claims disjointness across all three splits;
in fact only train<->test holds (train<->val shares 9 BioProjects, val<->test
shares 2).

So this script does three things v7 did not:

  G3  asserts disjointness in code, on every pair, and refuses to write on failure
  G4  reports per-fold class balance BEFORE training, so a degenerate split
      (v7's `community_type` was 98 % `oral` in test) is caught up front
  G5  marks classes that occur in only one BioProject as not evaluable
      out-of-project -- they stay in training but leave the macro metrics

It also replaces v7's hand-tuned rare-label forcing with plain
`StratifiedGroupKFold`, so the Methods sentence is one line and no class is
placed by hand.

Only runs that actually have features can be partitioned: the v7 matrix covers
train+test, and validation runs have per-sample vectors under
`results/validation_vectors_v7/`. Runs without features are excluded and counted.

Inputs
------
data/v9_labels_prepartition/{train,test,val}_metadata.tsv   (corrected targets)
data/matrices/matrix_v7_3190/unitigs.frac.mat  (which runs have matrix features)
results/validation_vectors_v7/                 (per-sample vectors)

Outputs (data/splits_v9/)
-------
{train,val,test}_accessions.txt, {train,val,test}_metadata.tsv
dev_folds.tsv            fold assignment for cross-validation
class_eligibility.tsv    per class: runs, BioProjects, evaluable yes/no
fold_report.txt          G4 balance report
split_config.json        seed, commit, counts, assertion log
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TARGETS = ["community_type", "feature", "sample_host", "material"]
GROUP_COL = "archive_project"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def runs_with_features(matrix_path: Path, vector_dir: Path) -> set:
    """Runs we can actually build a feature vector for."""
    from diana.data.loader import MatrixLoader

    have = set()
    if matrix_path.exists():
        _, ids, _ = MatrixLoader(str(matrix_path)).load()
        have |= {str(i) for i in ids}
        logger.info("matrix contributes %d runs", len(have))
    if vector_dir.exists():
        vec = {p.parent.name for p in vector_dir.glob("*/*_unitig_fraction.txt")
               if p.stat().st_size > 0}
        # An all-zero vector is unusable (truncated FASTQ, or reads shorter than k).
        usable = set()
        for acc in vec:
            f = vector_dir / acc / f"{acc}_unitig_fraction.txt"
            try:
                if np.loadtxt(f).sum() > 0:
                    usable.add(acc)
            except Exception:
                continue
        logger.info("vectors contribute %d runs (%d dropped as all-zero)",
                    len(usable), len(vec) - len(usable))
        have |= usable
    return have


def stratify_key(df: pd.DataFrame, min_count: int) -> np.ndarray:
    """Composite stratification key, with rare strata collapsed.

    StratifiedGroupKFold needs every class to have at least n_splits members, so
    strata below the threshold are pooled into one bucket rather than dropped.
    """
    key = (df["community_type"].fillna("_").astype(str) + "|" +
           df["feature"].fillna("_").astype(str) + "|" +
           df["material"].fillna("_").astype(str))
    vc = key.value_counts()
    return key.where(key.map(vc) >= min_count, "__rare__").to_numpy()


def assert_disjoint(parts: dict, groups: pd.Series, log: list) -> None:
    """G3. Every pair of splits must share no BioProject. Fail loudly."""
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga = set(groups.loc[parts[a]])
            gb = set(groups.loc[parts[b]])
            shared = ga & gb
            msg = f"{a} vs {b}: {len(ga)} / {len(gb)} BioProjects, shared={len(shared)}"
            log.append(msg)
            logger.info("  %s", msg)
            if shared:
                raise AssertionError(
                    f"{a} and {b} share {len(shared)} BioProject(s): "
                    f"{sorted(shared)[:5]}. Refusing to write a split that leaks."
                )


def class_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """G5. A class needs >=2 BioProjects to be evaluable out-of-project."""
    rows = []
    for t in TARGETS:
        sub = df[df[t].notna()]
        for cls, g in sub.groupby(t):
            n_bp = g[GROUP_COL].nunique()
            rows.append({"target": t, "class": cls, "n_runs": len(g),
                         "n_bioprojects": n_bp, "evaluable": bool(n_bp >= 2)})
    return pd.DataFrame(rows).sort_values(["target", "n_runs"], ascending=[True, False])


def fold_report(parts: dict, df: pd.DataFrame, elig: pd.DataFrame) -> str:
    """G4. Balance per split and target, printed before any training happens."""
    out = ["Fold balance (G4) — read this before training.", ""]
    ok = {(r.target, r["class"]) for _, r in elig.iterrows() if r.evaluable}
    for name, idx in parts.items():
        sub = df.loc[idx]
        out.append(f"{name}: {len(sub)} runs, {sub[GROUP_COL].nunique()} BioProjects")
        for t in TARGETS:
            s = sub[t].dropna()
            if not len(s):
                out.append(f"    {t:16s} (none)")
                continue
            maj = s.value_counts().iloc[0] / len(s)
            n_ev = s.isin([c for (tt, c) in ok if tt == t]).sum()
            flag = "   <-- DEGENERATE" if maj > 0.90 else ""
            out.append(f"    {t:16s} n={len(s):5d}  classes={s.nunique():3d}  "
                       f"majority={maj:6.1%}  evaluable-class runs={n_ev:5d}{flag}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data/v9_labels_prepartition")
    ap.add_argument("--matrix", type=Path,
                    default=PROJECT_ROOT / "data/matrices/matrix_v7_3190/unitigs.frac.mat")
    ap.add_argument("--vector-dir", type=Path,
                    default=PROJECT_ROOT / "results/validation_vectors_v7")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/splits_v9")
    ap.add_argument("--n-holdout-splits", type=int, default=7,
                    help="1/n of BioProjects become the frozen holdout (7 -> ~14 %%)")
    ap.add_argument("--n-dev-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-stratum", type=int, default=10)
    ap.add_argument("--allow-missing-features", action="store_true",
                    help="partition all runs, even those without features (testing only)")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(args.metadata_dir / f"{s}_metadata.tsv", sep="\t")
                    for s in ("train", "test", "val")], ignore_index=True)
    df = df.drop_duplicates("Run_accession").reset_index(drop=True)
    logger.info("pooled metadata: %d runs", len(df))

    if not args.allow_missing_features:
        have = runs_with_features(args.matrix, args.vector_dir)
        before = len(df)
        df = df[df.Run_accession.isin(have)].reset_index(drop=True)
        logger.info("runs with features: %d (excluded %d without)", len(df), before - len(df))

    if df[GROUP_COL].isna().any():
        raise ValueError(f"{int(df[GROUP_COL].isna().sum())} runs have no {GROUP_COL}; "
                         "cannot build a BioProject-disjoint split.")

    groups = df[GROUP_COL].astype(str)
    y = stratify_key(df, args.min_stratum)
    logger.info("%d BioProjects, %d strata", groups.nunique(), len(set(y)))

    # Frozen holdout: one fold of a grouped split over the whole corpus.
    sgkf = StratifiedGroupKFold(n_splits=args.n_holdout_splits, shuffle=True,
                                random_state=args.seed)
    rest_idx, hold_idx = next(iter(sgkf.split(df, y, groups=groups)))

    # Development folds over what remains; fold 0 becomes the canonical val.
    rest = df.iloc[rest_idx].reset_index(drop=True)
    rg = rest[GROUP_COL].astype(str)
    ry = stratify_key(rest, args.min_stratum)
    dev = StratifiedGroupKFold(n_splits=args.n_dev_folds, shuffle=True,
                               random_state=args.seed)
    fold_of = pd.Series(-1, index=rest.index, dtype=int)
    for k, (_, vi) in enumerate(dev.split(rest, ry, groups=rg)):
        fold_of.iloc[vi] = k

    val_local = rest.index[fold_of == 0]
    train_local = rest.index[fold_of != 0]
    parts = {
        "train": df.index[df.Run_accession.isin(rest.loc[train_local, "Run_accession"])],
        "val": df.index[df.Run_accession.isin(rest.loc[val_local, "Run_accession"])],
        "test": df.index[hold_idx],
    }

    log: list = []
    logger.info("G3: asserting BioProject disjointness on every pair")
    assert_disjoint(parts, groups, log)

    elig = class_eligibility(df)
    n_bad = int((~elig.evaluable).sum())
    logger.info("G5: %d of %d classes occur in a single BioProject", n_bad, len(elig))

    # G9-style check: every evaluable class must be learnable from training.
    oov = {}
    tr = df.loc[parts["train"]]
    for t in TARGETS:
        seen = set(tr[t].dropna())
        for name in ("val", "test"):
            s = df.loc[parts[name], t].dropna()
            oov[f"{name}_{t}"] = int((~s.isin(seen)).sum())
    logger.info("out-of-vocabulary runs per split/target (R3.5): %s",
                {k: v for k, v in oov.items() if v})

    report = fold_report(parts, df, elig)
    print("\n" + report)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, idx in parts.items():
        sub = df.loc[idx].sort_values("Run_accession")
        (args.output / f"{name}_accessions.txt").write_text(
            "\n".join(sub.Run_accession) + "\n")
        sub.to_csv(args.output / f"{name}_metadata.tsv", sep="\t", index=False)
    rest.assign(fold=fold_of.values)[["Run_accession", GROUP_COL, "fold"]].to_csv(
        args.output / "dev_folds.tsv", sep="\t", index=False)
    elig.to_csv(args.output / "class_eligibility.tsv", sep="\t", index=False)
    (args.output / "fold_report.txt").write_text(report + "\n")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    json.dump({
        "version": "v9",
        "method": "StratifiedGroupKFold on archive_project; disjointness asserted in code",
        "group_column": GROUP_COL,
        "seed": args.seed,
        "n_holdout_splits": args.n_holdout_splits,
        "n_dev_folds": args.n_dev_folds,
        "code_commit": commit,
        "counts": {k: int(len(v)) for k, v in parts.items()},
        "bioprojects": {k: int(groups.loc[v].nunique()) for k, v in parts.items()},
        "classes_single_bioproject": n_bad,
        "out_of_vocabulary": oov,
        "assertion_log": log,
    }, open(args.output / "split_config.json", "w"), indent=2)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
