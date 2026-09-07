#!/usr/bin/env python
"""Audit the v7 label space (PROJECT.md P1).

Question answered
-----------------
The v7 labels faithfully reproduce AncientMetagenomeDir (see audit_v7_labels.py),
so why can macro-F1 not be good? This quantifies the four *target-definition*
problems that cap it, independently of the model:

  1. `community_type` concatenates two disjoint AMD vocabularies — the
     host-associated `community_type` field and the environmental `feature` field.
  2. `sample_host` encodes "no host" as a class, so the head is partly a
     host-vs-environment detector and its accuracy is inflated by free wins.
  3. `material` has a long tail of classes with almost no support, and most of the
     label space has zero support in the test split.
  4. The library-row deduplication tie-break is order-dependent, so a run's label
     is not reproducible.

Inputs
------
data/metadata/AncientMetagenomeDir-v26.03.0/*.tsv
data/splits_v7/{train,test,val}_metadata.tsv
results/training_bioproject_v7/label_encoders.json
results/test_evaluation_bioproject_v7/test_predictions.tsv   (optional)

Outputs
-------
results/label_space_audit/label_space_audit.json
results/label_space_audit/class_support.tsv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMD_DIR = PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"
SPLITS = PROJECT_ROOT / "data/splits_v7"
OUT = PROJECT_ROOT / "results/label_space_audit"

LABEL_COLS = ["community_type", "sample_host", "material", "sample_type"]
TARGETS = ["community_type", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_amd_merged() -> pd.DataFrame:
    """Reproduce the merge in scripts/data_prep/04_build_v7_metadata.py."""
    hl = pd.read_csv(AMD_DIR / "ancientmetagenome-hostassociated_libraries.tsv", sep="\t", low_memory=False)
    hs = pd.read_csv(AMD_DIR / "ancientmetagenome-hostassociated_samples.tsv", sep="\t", low_memory=False)
    el = pd.read_csv(AMD_DIR / "ancientmetagenome-environmental_libraries.tsv", sep="\t", low_memory=False)
    es = pd.read_csv(AMD_DIR / "ancientmetagenome-environmental_samples.tsv", sep="\t", low_memory=False)

    host = hl.merge(
        hs[["project_name", "sample_name", "community_type", "sample_host", "material",
            "sample_age", "latitude", "longitude", "geo_loc_name"]],
        on=["project_name", "sample_name"], how="left")
    env = el.merge(
        es[["project_name", "sample_name", "feature", "material",
            "sample_age", "latitude", "longitude", "geo_loc_name"]],
        on=["project_name", "sample_name"], how="left")
    host["sample_type"] = "ancient_metagenome"
    env["sample_type"] = "ancient_metagenome"
    env["community_type"] = env["feature"]
    env["sample_host"] = None
    merged = pd.concat([host, env], ignore_index=True)
    return merged.rename(columns={"archive_data_accession": "Run_accession"})


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """The exact deduplication used by 04_build_v7_metadata.py."""
    df = df.copy()
    df["_c"] = df[LABEL_COLS].notna().sum(axis=1)
    return (df.sort_values("_c", ascending=False)
              .drop_duplicates(subset="Run_accession", keep="first")
              .set_index("Run_accession")[LABEL_COLS])


def audit_dedup_stability(merged: pd.DataFrame, n_perm: int = 10) -> dict:
    """How many runs change label when the input row order changes?"""
    ref = dedup(merged)
    unstable: set[str] = set()
    per_perm = []
    for seed in range(n_perm):
        out = dedup(merged.sample(frac=1.0, random_state=seed)).reindex(ref.index)
        changed = ref.index[(out.astype(str) != ref.astype(str)).any(axis=1)]
        per_perm.append(len(changed))
        unstable.update(changed)

    dup = merged[merged.duplicated("Run_accession", keep=False)]
    tmp = dup.copy()
    tmp["_c"] = tmp[LABEL_COLS].notna().sum(axis=1)
    tied = disagreeing = 0
    for _, g in tmp.groupby("Run_accession"):
        top = g[g._c == g._c.max()]
        if len(top) > 1:
            tied += 1
            if top[LABEL_COLS].astype(str).drop_duplicates().shape[0] > 1:
                disagreeing += 1
    return {
        "runs_total": int(merged.Run_accession.nunique()),
        "runs_with_multiple_library_rows": int(dup.Run_accession.nunique()),
        "runs_with_tied_completeness": tied,
        "runs_where_tied_rows_disagree": disagreeing,
        "runs_unstable_across_permutations": len(unstable),
        "changed_per_permutation": per_perm,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    merged = load_amd_merged()
    splits = {s: pd.read_csv(SPLITS / f"{s}_metadata.tsv", sep="\t") for s in ("train", "test", "val")}

    hs = pd.read_csv(AMD_DIR / "ancientmetagenome-hostassociated_samples.tsv", sep="\t", low_memory=False)
    es = pd.read_csv(AMD_DIR / "ancientmetagenome-environmental_samples.tsv", sep="\t", low_memory=False)
    host_vocab = set(hs["community_type"].dropna().unique())
    env_vocab = set(es["feature"].dropna().unique())

    report: dict = {
        "split_sizes": {s: int(len(d)) for s, d in splits.items()},
        "community_type_vocabularies": {
            "host_associated_terms": sorted(host_vocab),
            "environmental_terms": sorted(env_vocab),
            "overlap": sorted(host_vocab & env_vocab),
            "runs_by_source": {
                s: {"host_vocab": int(d["community_type"].isin(host_vocab).sum()),
                    "env_vocab": int(d["community_type"].isin(env_vocab).sum())}
                for s, d in splits.items()},
        },
        "missingness": {
            t: {s: int(d[t].isna().sum()) for s, d in splits.items()}
            for t in TARGETS + ["sample_age", "latitude", "longitude"]},
        "dedup_stability": audit_dedup_stability(merged),
    }

    # Class support and the ceiling that zero-support classes impose.
    rows = []
    label_space: dict = {}
    for t in TARGETS:
        train_cls = set(splits["train"][t].dropna().unique())
        test_cls = set(splits["test"][t].dropna().unique())
        label_space[t] = {
            "train_label_space": len(train_cls),
            "classes_with_test_support": len(train_cls & test_cls),
            "full_space_macro_f1_ceiling": round(len(train_cls & test_cls) / len(train_cls), 4),
        }
        allv = pd.concat([splits[s][t] for s in splits])
        for cls, tot in allv.value_counts().items():
            rows.append({
                "target": t, "class": cls,
                **{s: int((splits[s][t] == cls).sum()) for s in splits},
                "total": int(tot),
            })
    report["label_space"] = label_space
    pd.DataFrame(rows).to_csv(OUT / "class_support.tsv", sep="\t", index=False)

    # The null class inflates sample_host accuracy; quantify from real predictions.
    preds = PROJECT_ROOT / "results/test_evaluation_bioproject_v7/test_predictions.tsv"
    if preds.exists():
        df = pd.read_csv(preds, sep="\t")
        yt, yp = df["sample_host_true"], df["sample_host_pred"]
        real = yt.notna()
        report["sample_host_null_inflation"] = {
            "test_rows": int(len(df)),
            "rows_with_real_host": int(real.sum()),
            "accuracy_including_null_class": round(float((yt.astype(str) == yp.astype(str)).mean()), 4),
            "accuracy_on_real_hosts_only": round(
                float((yt[real].astype(str) == yp[real].astype(str)).mean()), 4),
            "free_correct_from_null_class": int((yt.isna() & yp.isna()).sum()),
        }

    with open(OUT / "label_space_audit.json", "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Wrote %s", OUT / "label_space_audit.json")
    logger.info("Wrote %s", OUT / "class_support.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
