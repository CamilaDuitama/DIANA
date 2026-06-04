#!/usr/bin/env python3
"""
Create BioProject-disjoint train/test splits for the DIANA dataset.

MOTIVATION (Reviewer concern):
    The original stratified random split (01_create_splits.py) assigns individual
    runs to train/test without regard to BioProject identity. Since 76 out of 77
    test BioProjects also appear in training, runs that share extraction protocol,
    library prep, sequencing platform, and possibly the same physical specimen end
    up on both sides of the split. BioProject identity is a known dominant
    confounder in microbiome ML (Wirbel 2021, Thomas 2019, Pasolli 2016).

APPROACH:
    1. Group all samples by BioProject.
    2. Treat each BioProject as an atomic unit — it goes entirely to train or test.
    3. Stratify BioProject assignment by each BioProject's majority community_type,
       so the class distribution is approximately preserved.
    4. Target ~15% of total *samples* (not BioProjects) in test.
    5. Singleton BioProjects (only 1 sample) all go to training to preserve rare
       class representation.
    6. Optionally filter a validation set to remove any samples whose BioProject
       also appears in train or test.

OUTPUT:
    data/splits_bioproject/
        train_ids.txt              Sample accessions for training
        test_ids.txt               Sample accessions for test
        train_metadata.tsv         Filtered metadata for training
        test_metadata.tsv          Filtered metadata for test
        validation_ids.txt         Validation accessions (BioProject-disjoint)
        validation_metadata.tsv    Filtered validation metadata
        split_config.json          Parameters and statistics

USAGE:
    mamba run -p ./env python scripts/data_prep/02_create_bioproject_disjoint_splits.py
    mamba run -p ./env python scripts/data_prep/02_create_bioproject_disjoint_splits.py \
        --metadata paper/metadata/train_metadata.tsv paper/metadata/test_metadata.tsv \
        --validation paper/metadata/validation_metadata.tsv \
        --output data/splits_bioproject \
        --test-size 0.15 \
        --random-state 42
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_bioproject_profile(df: pd.DataFrame,
                                bioproject_col: str,
                                label_col: str) -> pd.DataFrame:
    """
    For each BioProject compute: size, majority label, and label distribution.

    Returns a DataFrame indexed by BioProject with columns:
        n_samples, majority_label, label_entropy
    """
    records = []
    for bp, grp in df.groupby(bioproject_col):
        counts = grp[label_col].value_counts()
        majority = counts.idxmax()
        probs = counts / counts.sum()
        entropy = -(probs * np.log2(probs + 1e-12)).sum()
        records.append({
            bioproject_col: bp,
            "n_samples": len(grp),
            "majority_label": majority,
            "label_entropy": entropy,
        })
    return pd.DataFrame(records).set_index(bioproject_col)


def stratified_bioproject_split(df: pd.DataFrame,
                                 bioproject_col: str,
                                 label_col: str,
                                 test_size: float = 0.15,
                                 random_state: int = 42) -> tuple[list, list]:
    """
    Assign BioProjects to train or test such that:
      - No BioProject appears in both splits (disjoint by design)
      - Approximately `test_size` fraction of total samples end up in test
      - BioProject assignment is stratified by majority community_type
      - Singleton BioProjects always go to train

    Returns:
        train_accessions, test_accessions  (lists of Run_accession strings)
    """
    rng = np.random.default_rng(random_state)

    # Group samples by BioProject
    bp_groups: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        bp_groups[row[bioproject_col]].append(row["Run_accession"])

    # Build BioProject profile
    profile = compute_bioproject_profile(df, bioproject_col, label_col)

    total_samples = len(df)
    target_test_samples = int(total_samples * test_size)

    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total BioProjects: {len(bp_groups)}")
    logger.info(f"Target test samples: {target_test_samples} (~{test_size*100:.0f}%)")

    # Separate singleton BioProjects (always train)
    singleton_bps = profile[profile["n_samples"] == 1].index.tolist()
    multi_bps = profile[profile["n_samples"] > 1].index.tolist()
    logger.info(f"  Singleton BioProjects (→ train only): {len(singleton_bps)}")
    logger.info(f"  Multi-sample BioProjects (eligible for test): {len(multi_bps)}")

    # Sort multi-sample BioProjects by majority label for stratification
    multi_profile = profile.loc[multi_bps].copy()
    labels = sorted(multi_profile["majority_label"].unique())

    test_bp_set: set[str] = set()
    test_sample_count = 0

    # Stratify: for each majority-label group, shuffle and greedily pick BioProjects
    # for test until we reach the per-group quota
    for label in labels:
        label_bps = multi_profile[multi_profile["majority_label"] == label].index.tolist()
        label_sample_count = multi_profile.loc[label_bps, "n_samples"].sum()
        label_quota = int(label_sample_count * test_size)

        # Shuffle deterministically
        shuffled = rng.permutation(label_bps).tolist()

        accumulated = 0
        for bp in shuffled:
            if accumulated >= label_quota:
                break
            # Skip if adding this BioProject would exceed quota by more than its size
            bp_size = profile.loc[bp, "n_samples"]
            if accumulated + bp_size > label_quota * 1.5:
                continue
            test_bp_set.add(bp)
            accumulated += bp_size
            test_sample_count += bp_size

    # Log BioProject assignment
    train_bp_set = set(multi_bps) - test_bp_set
    train_bp_set.update(singleton_bps)

    logger.info(f"\nBioProject assignment:")
    logger.info(f"  Train BioProjects: {len(train_bp_set)}")
    logger.info(f"  Test BioProjects:  {len(test_bp_set)}")
    logger.info(f"  Overlap (must be 0): {len(train_bp_set & test_bp_set)}")
    assert len(train_bp_set & test_bp_set) == 0, "BioProject leak detected!"

    # Collect accessions
    train_accessions = []
    test_accessions = []
    for bp, accs in bp_groups.items():
        if bp in test_bp_set:
            test_accessions.extend(accs)
        else:
            train_accessions.extend(accs)

    logger.info(f"\nSample counts:")
    logger.info(f"  Train: {len(train_accessions)} ({len(train_accessions)/total_samples*100:.1f}%)")
    logger.info(f"  Test:  {len(test_accessions)} ({len(test_accessions)/total_samples*100:.1f}%)")

    return train_accessions, test_accessions


def report_class_distribution(df: pd.DataFrame,
                               split_name: str,
                               label_cols: list[str]):
    """Log class distribution for each task label."""
    logger.info(f"\n{split_name} class distributions:")
    for col in label_cols:
        counts = df[col].value_counts()
        fracs = (counts / len(df) * 100).round(1)
        logger.info(f"  {col}:")
        for cls, n in counts.items():
            logger.info(f"    {cls}: {n} ({fracs[cls]}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create BioProject-disjoint train/test splits"
    )
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=["paper/metadata/train_metadata.tsv", "paper/metadata/test_metadata.tsv"],
        help="Path(s) to train+test metadata TSV(s). Multiple files are concatenated.",
    )
    parser.add_argument(
        "--validation",
        default="paper/metadata/validation_metadata.tsv",
        help="Path to validation metadata TSV to filter for BioProject disjointness.",
    )
    parser.add_argument(
        "--output",
        default="data/splits_bioproject",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of samples for test set (default: 0.15)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--bioproject-col",
        default="BioProject",
        help="Column name for BioProject (default: BioProject)",
    )
    parser.add_argument(
        "--stratify-col",
        default="community_type",
        help="Column to stratify BioProject assignment by (default: community_type)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load train+test metadata (one or more files concatenated)
    # -------------------------------------------------------------------------
    metadata_paths = [Path(p) for p in args.metadata]
    for p in metadata_paths:
        if not p.exists():
            logger.error(f"Metadata file not found: {p}")
            sys.exit(1)

    logger.info(f"Loading train+test metadata from: {', '.join(str(p) for p in metadata_paths)}")
    dfs = [pd.read_csv(p, sep="\t", low_memory=False) for p in metadata_paths]
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} samples ({', '.join(str(len(d)) for d in dfs)} from each file)")

    # Drop duplicate Run_accessions
    before = len(df)
    df = df.drop_duplicates(subset=["Run_accession"])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} duplicate Run_accession rows")

    # Drop rows with missing Run_accession
    before = len(df)
    df = df.dropna(subset=["Run_accession"])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} rows with missing Run_accession")

    # Fill missing BioProject with a unique placeholder (treat each as its own group)
    n_missing_bp = df[args.bioproject_col].isna().sum()
    if n_missing_bp > 0:
        logger.warning(
            f"{n_missing_bp} samples have missing BioProject — assigning unique "
            f"placeholder IDs (these will go to train as singletons)"
        )
        missing_mask = df[args.bioproject_col].isna()
        df.loc[missing_mask, args.bioproject_col] = (
            "UNKNOWN_" + df.loc[missing_mask, "Run_accession"]
        )

    # -------------------------------------------------------------------------
    # Create BioProject-disjoint split
    # -------------------------------------------------------------------------
    train_ids, test_ids = stratified_bioproject_split(
        df=df,
        bioproject_col=args.bioproject_col,
        label_col=args.stratify_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_df = df[df["Run_accession"].isin(train_ids)].copy()
    test_df = df[df["Run_accession"].isin(test_ids)].copy()

    # Verify disjointness
    overlap = set(train_ids) & set(test_ids)
    assert len(overlap) == 0, f"Sample leak detected: {len(overlap)} shared accessions"

    # Verify BioProject disjointness
    train_bps = set(train_df[args.bioproject_col].unique())
    test_bps = set(test_df[args.bioproject_col].unique())
    bp_overlap = train_bps & test_bps
    assert len(bp_overlap) == 0, f"BioProject leak: {bp_overlap}"
    logger.info(f"\n✓ Verified: 0 shared samples, 0 shared BioProjects")

    # Report class distributions
    label_cols = ["sample_type", "community_type", "sample_host", "material"]
    label_cols = [c for c in label_cols if c in df.columns]
    report_class_distribution(train_df, "Train", label_cols)
    report_class_distribution(test_df, "Test", label_cols)

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train_ids.txt").write_text("\n".join(train_ids))
    (output_dir / "test_ids.txt").write_text("\n".join(test_ids))
    logger.info(f"\nSaved split IDs to {output_dir}/{{train,test}}_ids.txt")

    train_df.to_csv(output_dir / "train_metadata.tsv", sep="\t", index=False)
    test_df.to_csv(output_dir / "test_metadata.tsv", sep="\t", index=False)
    logger.info(f"Saved metadata to {output_dir}/{{train,test}}_metadata.tsv")

    # Collect BioProject stats for the config
    train_bp_counts = train_df[args.bioproject_col].value_counts().to_dict()
    test_bp_counts = test_df[args.bioproject_col].value_counts().to_dict()

    config = {
        "method": "bioproject_disjoint",
        "description": (
            "BioProjects are assigned entirely to train or test — no BioProject "
            "appears in both splits. Addresses reviewer concern about BioProject "
            "identity as a dominant confounder in microbiome ML."
        ),
        "parameters": {
            "metadata": args.metadata,
            "validation": args.validation,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "bioproject_col": args.bioproject_col,
            "stratify_col": args.stratify_col,
        },
        "results": {
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_total": len(df),
            "actual_test_fraction": round(len(test_ids) / len(df), 4),
            "n_train_bioprojects": len(train_bps),
            "n_test_bioprojects": len(test_bps),
            "bioproject_overlap": 0,
        },
        "class_distributions": {},
    }
    for col in label_cols:
        config["class_distributions"][col] = {
            "train": train_df[col].value_counts().to_dict(),
            "test": test_df[col].value_counts().to_dict(),
        }

    with open(output_dir / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved split config to {output_dir}/split_config.json")

    # -------------------------------------------------------------------------
    # Filter validation set to BioProject-disjoint samples
    # -------------------------------------------------------------------------
    all_train_test_bps = train_bps | test_bps
    val_path = Path(args.validation)
    if val_path.exists():
        logger.info(f"\nLoading validation metadata from {val_path}")
        val_df = pd.read_csv(val_path, sep="\t", low_memory=False)
        logger.info(f"  Original validation set: {len(val_df)} samples")

        val_bp_col = args.bioproject_col
        if val_bp_col not in val_df.columns:
            logger.error(f"BioProject column '{val_bp_col}' not found in validation metadata")
        else:
            val_before = len(val_df)
            val_df_filtered = val_df[~val_df[val_bp_col].isin(all_train_test_bps)].copy()
            removed = val_before - len(val_df_filtered)
            logger.info(f"  Removed {removed} samples whose BioProject appears in train/test")
            logger.info(f"  Filtered validation set: {len(val_df_filtered)} samples")

            val_ids = val_df_filtered["Run_accession"].tolist()
            (output_dir / "validation_ids.txt").write_text("\n".join(val_ids))
            val_df_filtered.to_csv(output_dir / "validation_metadata.tsv", sep="\t", index=False)
            logger.info(f"  Saved to {output_dir}/validation_{{ids.txt,metadata.tsv}}")

            val_bps_before = set(val_df[val_bp_col].dropna().unique())
            val_bps_after = set(val_df_filtered[val_bp_col].dropna().unique())
            logger.info(f"  BioProjects: {len(val_bps_before)} → {len(val_bps_after)}")

            config["results"]["n_validation_original"] = val_before
            config["results"]["n_validation_filtered"] = len(val_df_filtered)
            config["results"]["n_validation_removed"] = removed
            config["results"]["n_validation_bioprojects"] = len(val_bps_after)

            # Verify triple disjointness
            assert len(train_bps & val_bps_after) == 0, "BioProject leak: train ∩ validation"
            assert len(test_bps & val_bps_after) == 0, "BioProject leak: test ∩ validation"
            logger.info("  ✓ Verified: 0 BioProject overlap between train, test, and validation")

            with open(output_dir / "split_config.json", "w") as f:
                json.dump(config, f, indent=2)
    else:
        logger.warning(f"Validation metadata not found at {val_path} — skipping validation filtering")

    logger.info("\n" + "=" * 60)
    logger.info("BIOPROJECT-DISJOINT SPLIT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Train:      {len(train_ids)} samples across {len(train_bps)} BioProjects")
    logger.info(f"  Test:       {len(test_ids)} samples across {len(test_bps)} BioProjects")
    if val_path.exists() and val_bp_col in val_df.columns:
        logger.info(f"  Validation: {len(val_df_filtered)} samples across {len(val_bps_after)} BioProjects (was {val_before})")
    logger.info(f"  Shared BioProjects (train/test): 0")
    logger.info(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
