#!/usr/bin/env python3
"""
Create BioProject-disjoint train/val/test splits with GUARANTEED LABEL COVERAGE.

FIXES V6 ISSUE:
    V6 splits had "plant tissue" and "soft tissue" missing from training because
    these labels only appeared in bioprojects that were randomly assigned to val/test.
    
SOLUTION (V7):
    1. Identify rare community_type labels (≤2 bioprojects) and force to training
    2. Stratify host and environmental samples SEPARATELY to prevent competition
    3. Remove singleton/near-singleton classes (< 5 samples) from training
    4. Accept rare material labels as zero-shot cases (not forced to training)
    5. Generate label distribution report automatically

KEY IMPROVEMENTS:
    - Only community_type rare labels forced to training (material can be zero-shot)
    - Host and env samples stratified independently (no quota competition)
    - Tighter quota multiplier (1.2× vs 1.5×) prevents large bioproject dominance
    - Singleton removal ensures reliable class learning
    - Integrated label distribution reporting

USAGE:
    python scripts/data_prep/03_create_bioproject_splits_v7_fixed.py \
        --output data/splits_v7 \
        --train-size 0.70 \
        --val-size 0.15 \
        --test-size 0.15 \
        --random-state 42 \
        --min-samples-per-class 5
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_bioproject_profile(df: pd.DataFrame,
                                bioproject_col: str,
                                label_col: str,
                                accession_col: str) -> pd.DataFrame:
    """
    For each BioProject compute: size, majority label, and label distribution.
    """
    records = []
    for bp, grp in df.groupby(bioproject_col):
        counts = grp[label_col].value_counts()
        if len(counts) == 0:
            continue
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


def find_rare_labels(df: pd.DataFrame,
                     bioproject_col: str,
                     label_col: str,
                     max_bioprojects: int = 2) -> Dict[str, List[str]]:
    """
    Find labels that appear in very few bioprojects (≤ max_bioprojects).
    
    ONLY use for community_type — material rare labels should be zero-shot cases.
    
    Returns:
        dict mapping label -> list of bioprojects containing that label
    """
    label_to_bioprojects = defaultdict(set)
    
    for bp, grp in df.groupby(bioproject_col):
        labels = grp[label_col].dropna().unique()
        for label in labels:
            label_to_bioprojects[label].add(bp)
    
    # Filter to rare labels
    rare_labels = {
        label: list(bps) 
        for label, bps in label_to_bioprojects.items() 
        if len(bps) <= max_bioprojects
    }
    
    return rare_labels


def stratified_bioproject_split_with_label_coverage(
        df: pd.DataFrame,
        bioproject_col: str,
        stratify_col: str,
        label_cols: List[str],
        accession_col: str,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
        min_samples_per_class: int = 5) -> Tuple[List[str], List[str], List[str]]:
    """
    Assign BioProjects to train/val/test ensuring:
      1. No BioProject appears in multiple splits (disjoint by design)
      2. community_type rare labels in TRAINING (material rare labels OK as zero-shot)
      3. Approximately correct split proportions
      4. Stratified SEPARATELY for host and environmental samples
      5. Classes with < min_samples_per_class removed from training
    
    Strategy:
      - Identify rare community_type labels (appear in ≤2 bioprojects)
      - Force at least one bioproject with each rare community_type → TRAINING
      - Stratify host and environmental samples separately then combine
      - Remove singleton/near-singleton classes after split
    
    Returns:
        train_accessions, val_accessions, test_accessions (after filtering)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    
    rng = np.random.default_rng(random_state)

    # Group samples by BioProject
    bp_groups: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        bp_groups[row[bioproject_col]].append(row[accession_col])

    total_samples = len(df)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total BioProjects: {len(bp_groups)}")
    logger.info(f"Target split: train={train_size:.0%}, val={val_size:.0%}, test={test_size:.0%}")
    
    # =========================================================================
    # STEP 1: Identify rare community_type labels ONLY (not material)
    # =========================================================================
    forced_train_bps: Set[str] = set()
    
    if stratify_col in df.columns:
        # Bug Fix 3: Use host-only df for rare community_type label search
        host_only_df = df[df['sample_type'] == 'host-associated']
        rare_labels = find_rare_labels(host_only_df, bioproject_col, stratify_col, max_bioprojects=2)
        if rare_labels:
            logger.info(f"\n🔍 {stratify_col}: Found {len(rare_labels)} rare labels (≤2 bioprojects)")
            
            # Build profile for sizing
            profile_for_sizing = compute_bioproject_profile(df, bioproject_col, stratify_col, accession_col)
            
            for label, bps in sorted(rare_labels.items()):
                # Pick the largest bioproject with this label for training
                bp_sizes = [(bp, profile_for_sizing.loc[bp, "n_samples"]) if bp in profile_for_sizing.index else (bp, 1) 
                           for bp in bps]
                if bp_sizes:
                    largest_bp = max(bp_sizes, key=lambda x: x[1])[0]
                    if largest_bp not in forced_train_bps:
                        forced_train_bps.add(largest_bp)
                        logger.info(f"  ✓ {largest_bp} → TRAIN (ensures '{label}' in training)")
    
    logger.info(f"\nTotal forced-train bioprojects: {len(forced_train_bps)}")
    
    # =========================================================================
    # STEP 2: Stratify HOST and ENVIRONMENTAL samples SEPARATELY
    # =========================================================================
    logger.info(f"\n📊 Stratifying host and environmental samples separately...")
    
    # Separate host and env
    host_df = df[df['sample_type'] == 'host-associated']
    env_df = df[df['sample_type'] == 'environmental']
    
    logger.info(f"  Host-associated: {len(host_df)} samples")
    logger.info(f"  Environmental: {len(env_df)} samples")
    
    def stratify_sample_type(sub_df: pd.DataFrame, 
                             sample_type_name: str) -> Tuple[Set[str], Set[str]]:
        """Stratify one sample type (host or env) and return val/test bioproject sets."""
        if len(sub_df) == 0:
            return set(), set()
        
        # Build profile for this sample type
        profile = compute_bioproject_profile(sub_df, bioproject_col, stratify_col, accession_col)
        
        # Separate singleton and multi-sample bioprojects
        singleton_bps = profile[profile["n_samples"] == 1].index.tolist()
        multi_bps = profile[profile["n_samples"] > 1].index.tolist()
        
        # Remove forced-train bioprojects from eligible pool
        multi_bps = [bp for bp in multi_bps if bp not in forced_train_bps]
        
        logger.info(f"\n  {sample_type_name}:")
        logger.info(f"    Singleton BioProjects (→ train): {len(singleton_bps)}")
        logger.info(f"    Eligible for val/test: {len(multi_bps)}")
        
        if len(multi_bps) == 0:
            logger.info(f"    No multi-sample bioprojects available for val/test")
            return set(), set()
        
        multi_profile = profile.loc[multi_bps].copy()
        labels = sorted(multi_profile["majority_label"].unique())
        
        val_bp_set: Set[str] = set()
        test_bp_set: Set[str] = set()
        
        for label in labels:
            label_bps = multi_profile[multi_profile["majority_label"] == label].index.tolist()
            if not label_bps:
                continue
            
            label_sample_count = multi_profile.loc[label_bps, "n_samples"].sum()
            label_val_quota = int(label_sample_count * val_size)
            label_test_quota = int(label_sample_count * test_size)
            
            # Shuffle deterministically
            shuffled = rng.permutation(label_bps).tolist()
            
            # Assign to val (tightened multiplier: 1.2× instead of 1.5×)
            accumulated_val = 0
            for bp in shuffled[:]:
                if accumulated_val >= label_val_quota:
                    break
                bp_size = profile.loc[bp, "n_samples"]
                if accumulated_val + bp_size <= label_val_quota * 1.2:
                    val_bp_set.add(bp)
                    accumulated_val += bp_size
                    shuffled.remove(bp)
            
            # Assign to test (tightened multiplier: 1.2× instead of 1.5×)
            accumulated_test = 0
            for bp in shuffled[:]:
                if accumulated_test >= label_test_quota:
                    break
                bp_size = profile.loc[bp, "n_samples"]
                if accumulated_test + bp_size <= label_test_quota * 1.2:
                    test_bp_set.add(bp)
                    accumulated_test += bp_size
                    shuffled.remove(bp)
        
        return val_bp_set, test_bp_set
    
    # Stratify each sample type
    host_val_bps, host_test_bps = stratify_sample_type(host_df, "Host-associated")
    env_val_bps, env_test_bps = stratify_sample_type(env_df, "Environmental")
    
    # Combine
    val_bp_set = host_val_bps | env_val_bps
    test_bp_set = host_test_bps | env_test_bps
    
    # Get all bioprojects
    all_bps = set(bp_groups.keys())
    singleton_bps = [bp for bp in all_bps 
                     if bp not in val_bp_set and bp not in test_bp_set and bp not in forced_train_bps
                     and len(bp_groups[bp]) == 1]

    # =========================================================================
    # STEP 3: Assign remaining bioprojects to training
    # =========================================================================
    train_bp_set = all_bps - val_bp_set - test_bp_set

    logger.info(f"\n📊 Initial BioProject assignment:")
    logger.info(f"  Train BioProjects: {len(train_bp_set)}")
    logger.info(f"  Val BioProjects:   {len(val_bp_set)}")
    logger.info(f"  Test BioProjects:  {len(test_bp_set)}")
    
    # Verify disjointness
    assert len(train_bp_set & val_bp_set) == 0, "Train/Val overlap!"
    assert len(train_bp_set & test_bp_set) == 0, "Train/Test overlap!"
    assert len(val_bp_set & test_bp_set) == 0, "Val/Test overlap!"
    logger.info(f"  ✓ Verified: All splits are disjoint")

    # =========================================================================
    # STEP 4: Collect accessions (before singleton removal)
    # =========================================================================
    train_accessions = []
    val_accessions = []
    test_accessions = []
    
    for bp, accs in bp_groups.items():
        if bp in train_bp_set:
            train_accessions.extend(accs)
        elif bp in val_bp_set:
            val_accessions.extend(accs)
        elif bp in test_bp_set:
            test_accessions.extend(accs)

    logger.info(f"\n📈 Sample counts (before singleton removal):")
    logger.info(f"  Train: {len(train_accessions)} ({len(train_accessions)/total_samples*100:.1f}%)")
    logger.info(f"  Val:   {len(val_accessions)} ({len(val_accessions)/total_samples*100:.1f}%)")
    logger.info(f"  Test:  {len(test_accessions)} ({len(test_accessions)/total_samples*100:.1f}%)")
    
    # =========================================================================
    # STEP 5: Mask singleton/near-singleton labels to NaN (keep samples)
    # =========================================================================
    logger.info(f"\n🧹 Masking labels with < {min_samples_per_class} samples per class to NaN...")
    
    # Bug Fix 2: Mask labels to NaN instead of removing samples
    # Create a working copy for masking
    df_masked = df.copy()
    total_masked = 0
    
    for col in label_cols:
        if col not in df.columns:
            continue
        
        # Bug Fix 1: Recompute train_df inside the loop
        train_df = df_masked[df_masked[accession_col].isin(train_accessions)]
        
        # Count samples per label in training
        label_counts = train_df[col].value_counts()
        singleton_labels = label_counts[label_counts < min_samples_per_class].index.tolist()
        
        if singleton_labels:
            logger.info(f"  {col}: Masking {len(singleton_labels)} labels with < {min_samples_per_class} samples")
            for label in singleton_labels:
                n_samples = label_counts[label]
                logger.info(f"    - '{label}': {n_samples} samples → MASKED to NaN")
                
                # Mask this label to NaN in training samples only
                mask = (df_masked[accession_col].isin(train_accessions)) & (df_masked[col] == label)
                df_masked.loc[mask, col] = np.nan
                total_masked += mask.sum()
    
    if total_masked > 0:
        logger.info(f"\n  Total label values masked: {total_masked}")
        logger.info(f"  Training samples retained: {len(train_accessions)} ({len(train_accessions)/total_samples*100:.1f}%)")
    else:
        logger.info(f"  ✅ No singleton classes found — all classes have ≥ {min_samples_per_class} samples")

    return train_accessions, val_accessions, test_accessions, df_masked if total_masked > 0 else df


def verify_label_coverage(df: pd.DataFrame,
                          train_ids: List[str],
                          val_ids: List[str],
                          test_ids: List[str],
                          label_cols: List[str],
                          accession_col: str) -> bool:
    """
    Verify label coverage across splits.
    
    NOTE: Missing labels in training are now logged as WARNINGS, not errors.
          Unseen classes are expected for zero-shot evaluation (e.g., rare materials).
    """
    logger.info(f"\n🔬 Verifying label coverage...")
    
    train_df = df[df[accession_col].isin(train_ids)]
    val_df = df[df[accession_col].isin(val_ids)]
    test_df = df[df[accession_col].isin(test_ids)]
    
    all_good = True
    for col in label_cols:
        if col not in df.columns:
            continue
            
        train_labels = set(train_df[col].dropna().unique())
        val_labels = set(val_df[col].dropna().unique())
        test_labels = set(test_df[col].dropna().unique())
        
        missing_in_train = (val_labels | test_labels) - train_labels
        
        if missing_in_train:
            logger.warning(f"  ⚠️  {col}: {len(missing_in_train)} labels missing from training (zero-shot scenario)")
            for label in sorted(missing_in_train):
                # Count occurrences
                n_val = (val_df[col] == label).sum()
                n_test = (test_df[col] == label).sum()
                logger.warning(f"      - '{label}': val={n_val}, test={n_test}")
            all_good = False
        else:
            logger.info(f"  ✅ {col}: All labels in training ({len(train_labels)} unique)")
    
    return all_good


def main():
    parser = argparse.ArgumentParser(
        description="Create BioProject-disjoint train/val/test splits with guaranteed label coverage"
    )
    parser.add_argument(
        "--metadata-host-lib",
        default="data/metadata/AncientMetagenomeDir-v26.03.0/ancientmetagenome-hostassociated_libraries.tsv",
        help="Path to host-associated libraries TSV",
    )
    parser.add_argument(
        "--metadata-host-samples",
        default="data/metadata/AncientMetagenomeDir-v26.03.0/ancientmetagenome-hostassociated_samples.tsv",
        help="Path to host-associated samples TSV",
    )
    parser.add_argument(
        "--metadata-env-lib",
        default="data/metadata/AncientMetagenomeDir-v26.03.0/ancientmetagenome-environmental_libraries.tsv",
        help="Path to environmental libraries TSV",
    )
    parser.add_argument(
        "--metadata-env-samples",
        default="data/metadata/AncientMetagenomeDir-v26.03.0/ancientmetagenome-environmental_samples.tsv",
        help="Path to environmental samples TSV",
    )
    parser.add_argument(
        "--logan-accessions",
        default="data/metadata/AncientMetagenomeDir-v26.03.0/all_amd_accessions.txt",
        help="Path to file with accessions that have Logan unitigs",
    )
    parser.add_argument(
        "--output",
        default="data/splits_v7",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.70,
        help="Fraction for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction for test (default: 0.15)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=5,
        help="Minimum samples per class in training (default: 5). Classes with fewer samples are removed.",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load and merge metadata
    # -------------------------------------------------------------------------
    logger.info(f"Loading metadata...")
    
    # Load libraries
    host_lib = pd.read_csv(args.metadata_host_lib, sep="\t", low_memory=False)
    env_lib = pd.read_csv(args.metadata_env_lib, sep="\t", low_memory=False)
    
    # Load samples (contains labels)
    host_samples = pd.read_csv(args.metadata_host_samples, sep="\t", low_memory=False)
    env_samples = pd.read_csv(args.metadata_env_samples, sep="\t", low_memory=False)
    
    # Merge libraries with samples on project_name and sample_name
    host_df = host_lib.merge(
        host_samples[['project_name', 'sample_name', 'community_type', 'sample_host', 'material', 'sample_age', 'latitude', 'longitude']],
        on=['project_name', 'sample_name'],
        how='left'
    )
    env_df = env_lib.merge(
        env_samples[['project_name', 'sample_name', 'feature', 'material', 'sample_age', 'latitude', 'longitude']],
        on=['project_name', 'sample_name'],
        how='left'
    )
    
    host_df['sample_type'] = 'host-associated'
    env_df['sample_type'] = 'environmental'
    
    # For environmental samples, copy feature → community_type for consistency
    env_df['community_type'] = env_df['feature']
    
    df = pd.concat([host_df, env_df], ignore_index=True)
    logger.info(f"Loaded {len(df)} total libraries ({len(host_df)} host + {len(env_df)} env)")

    # Filter to samples with Logan unitigs
    logan_accessions = set(open(args.logan_accessions).read().strip().split('\n'))
    logger.info(f"Loaded {len(logan_accessions)} Logan accessions")
    
    accession_col = "archive_data_accession"
    before = len(df)
    df = df[df[accession_col].isin(logan_accessions)]
    logger.info(f"Filtered to {len(df)} libraries with Logan unitigs (dropped {before - len(df)})")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[accession_col])
    if len(df) < before:
        logger.warning(f"Dropped {before - len(df)} duplicate {accession_col} rows")

    # Fill missing BioProject
    bioproject_col = "project_name"
    n_missing_bp = df[bioproject_col].isna().sum()
    if n_missing_bp > 0:
        logger.warning(f"{n_missing_bp} samples missing {bioproject_col} → assigning unique IDs")
        missing_mask = df[bioproject_col].isna()
        df.loc[missing_mask, bioproject_col] = "UNKNOWN_" + df.loc[missing_mask, accession_col]

    # -------------------------------------------------------------------------
    # Create splits with label coverage guarantee for community_type only
    # -------------------------------------------------------------------------
    label_cols = ["community_type", "sample_host", "material", "feature"]
    train_ids, val_ids, test_ids, df = stratified_bioproject_split_with_label_coverage(
        df=df,
        bioproject_col=bioproject_col,
        stratify_col="community_type",
        label_cols=label_cols,
        accession_col=accession_col,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        min_samples_per_class=args.min_samples_per_class,
    )

    train_df = df[df[accession_col].isin(train_ids)].copy()
    val_df = df[df[accession_col].isin(val_ids)].copy()
    test_df = df[df[accession_col].isin(test_ids)].copy()

    # -------------------------------------------------------------------------
    # Sanity check: warn if host-associated samples have environmental materials
    # -------------------------------------------------------------------------
    logger.info(f"\n🔍 Running sanity checks...")
    
    # Environmental material types that should not appear in host-associated samples
    env_materials = ['sediment', 'permafrost', 'lake sediment', 'soil', 'peat']
    
    host_with_env_material = df[
        (df['sample_type'] == 'host-associated') & 
        (df['material'].isin(env_materials))
    ]
    
    if len(host_with_env_material) > 0:
        logger.warning(f"  ⚠️  Found {len(host_with_env_material)} host-associated samples with environmental material types:")
        for mat in host_with_env_material['material'].value_counts().head(10).items():
            logger.warning(f"      - {mat[0]}: {mat[1]} samples")
    else:
        logger.info(f"  ✅ No host-associated samples with environmental materials")

    # -------------------------------------------------------------------------
    # Verify label coverage (warnings only, not errors)
    # -------------------------------------------------------------------------
    label_cols = ["community_type", "sample_host", "material", "feature"]
    all_labels_covered = verify_label_coverage(df, train_ids, val_ids, test_ids, label_cols, accession_col)
    
    if not all_labels_covered:
        logger.info(f"\n⚠️  Some labels missing from training (acceptable for zero-shot evaluation)")
    else:
        logger.info(f"\n✅ All labels present in training set!")

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train_accessions.txt").write_text("\n".join(train_ids))
    (output_dir / "val_accessions.txt").write_text("\n".join(val_ids))
    (output_dir / "test_accessions.txt").write_text("\n".join(test_ids))
    logger.info(f"\nSaved split IDs to {output_dir}/{{train,val,test}}_accessions.txt")

    # Create split report
    split_report = []
    for bp in df[bioproject_col].unique():
        bp_df = df[df[bioproject_col] == bp]
        bp_train = bp_df[bp_df[accession_col].isin(train_ids)]
        bp_val = bp_df[bp_df[accession_col].isin(val_ids)]
        bp_test = bp_df[bp_df[accession_col].isin(test_ids)]
        
        if len(bp_train) > 0:
            split = "train"
        elif len(bp_val) > 0:
            split = "val"
        elif len(bp_test) > 0:
            split = "test"
        else:
            split = "none"
        
        community_types = ", ".join(sorted(bp_df['community_type'].dropna().unique()))
        sample_hosts = ", ".join(sorted(bp_df['sample_host'].dropna().unique())[:5])  # Limit to 5
        materials = ", ".join(sorted(bp_df['material'].dropna().unique())[:5])
        
        split_report.append({
            'project_name': bp,
            'n_runs': len(bp_df),
            'community_types': community_types,
            'sample_hosts': sample_hosts,
            'materials': materials,
            'split': split
        })
    
    split_report_df = pd.DataFrame(split_report)
    split_report_df.to_csv(output_dir / "split_report.tsv", sep="\t", index=False)
    logger.info(f"Saved split report to {output_dir}/split_report.tsv")

    # -------------------------------------------------------------------------
    # Generate label distribution report
    # -------------------------------------------------------------------------
    logger.info(f"\n📊 Generating label distribution report...")
    
    total_samples = len(df)
    report_lines = []
    def log_report(line=""):
        """Print and save to report."""
        print(line)
        report_lines.append(line)
    
    log_report("=" * 80)
    log_report("V7 LABEL DISTRIBUTION REPORT")
    log_report("=" * 80)
    log_report()
    log_report(f"Train: {len(train_ids)} samples ({len(train_ids)/total_samples*100:.1f}%)")
    log_report(f"Val:   {len(val_ids)} samples ({len(val_ids)/total_samples*100:.1f}%)")
    log_report(f"Test:  {len(test_ids)} samples ({len(test_ids)/total_samples*100:.1f}%)")
    log_report()
    
    def is_continuous_variable(task: str, task_df: pd.DataFrame) -> bool:
        """Check if a task should be treated as continuous."""
        if task in ['sample_age', 'latitude', 'longitude']:
            return True
        return False
    
    def analyze_task(task_df: pd.DataFrame, task: str, sample_type_filter: str = None):
        """Analyze label distribution for a task."""
        if sample_type_filter:
            task_df = task_df[task_df['sample_type'] == sample_type_filter]
        
        if task not in task_df.columns:
            return
        
        log_report(f"\n{'='*80}")
        log_report(f"Task: {task.upper()}")
        if sample_type_filter:
            log_report(f"Sample Type: {sample_type_filter}")
        log_report(f"{'='*80}")
        
        continuous = is_continuous_variable(task, task_df)
        
        for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
            split_df = task_df[task_df[accession_col].isin(split_ids)]
            
            if len(split_df) == 0:
                continue
            
            valid = split_df[task].notna()
            n_valid = valid.sum()
            n_total = len(split_df)
            
            log_report(f"\n{split_name.upper()} ({n_total} samples, {n_valid} with labels)")
            log_report("-" * 80)
            
            if n_valid > 0:
                if continuous:
                    # Show statistics for continuous variables
                    values = split_df[split_df[task].notna()][task].astype(float)
                    
                    # Apply log10 transformation for sample_age to handle extreme outliers
                    if task == 'sample_age':
                        log_values = np.log10(values)
                        log_report(f"  Mean (log10):   {log_values.mean():12.2f}")
                        log_report(f"  Median (log10): {log_values.median():12.2f}")
                        log_report(f"  Std (log10):    {log_values.std():12.2f}")
                        log_report(f"  Min (log10):    {log_values.min():12.2f}  [{values.min():12.0f} years]")
                        log_report(f"  Q25 (log10):    {log_values.quantile(0.25):12.2f}")
                        log_report(f"  Q75 (log10):    {log_values.quantile(0.75):12.2f}")
                        log_report(f"  Max (log10):    {log_values.max():12.2f}  [{values.max():12.0f} years]")
                    else:
                        log_report(f"  Mean:   {values.mean():12.2f}")
                        log_report(f"  Median: {values.median():12.2f}")
                        log_report(f"  Std:    {values.std():12.2f}")
                        log_report(f"  Min:    {values.min():12.2f}")
                        log_report(f"  Q25:    {values.quantile(0.25):12.2f}")
                        log_report(f"  Q75:    {values.quantile(0.75):12.2f}")
                        log_report(f"  Max:    {values.max():12.2f}")
                    
                    log_report(f"\n  Total values: {len(values)}")
                else:
                    # Show value counts for categorical variables
                    value_counts = split_df[split_df[task].notna()][task].value_counts()
                    max_display = 20 if task in ['community_type', 'sample_host', 'material', 'feature'] else 10
                    
                    for i, (label, count) in enumerate(value_counts.head(max_display).items()):
                        pct = 100 * count / n_valid
                        log_report(f"  {i+1:2d}. {str(label):45s} {count:5d} ({pct:5.1f}%)")
                    
                    if len(value_counts) > max_display:
                        remaining = len(value_counts) - max_display
                        remaining_count = value_counts.iloc[max_display:].sum()
                        log_report(f"  ... +{remaining} more labels ({remaining_count} samples)")
                    
                    log_report(f"\n  Total unique labels: {len(value_counts)}")
    
    # Analyze tasks
    log_report("\n" + "█" * 80)
    log_report("HOST-ASSOCIATED SAMPLES")
    log_report("█" * 80)
    
    for task in ['community_type', 'sample_host', 'material', 'sample_age', 'latitude', 'longitude']:
        analyze_task(df, task, 'host-associated')
    
    log_report("\n\n" + "█" * 80)
    log_report("ENVIRONMENTAL SAMPLES")
    log_report("█" * 80)
    
    for task in ['feature', 'material', 'sample_age', 'latitude', 'longitude']:
        analyze_task(df, task, 'environmental')
    
    # -------------------------------------------------------------------------
    # File inventory section
    # -------------------------------------------------------------------------
    log_report("\n\n" + "█" * 80)
    log_report("FILE INVENTORY")
    log_report("█" * 80)
    log_report("\nThis section tracks available files vs. required files for each split.")
    log_report("Train/Test use unitig files; Validation uses raw FASTQ files.\n")
    
    # Check unitig files
    unitig_dir = Path("data/unitigs")
    if unitig_dir.exists():
        unitig_files = set()
        for ext in ['*.unitigs.fa', '*.unitigs.fasta', '*.unitigs.fa.gz', '*.unitigs.fasta.gz']:
            for f in unitig_dir.glob(ext):
                # Remove .unitigs.fa, .unitigs.fasta, .unitigs.fa.gz, or .unitigs.fasta.gz
                name = f.name
                for suffix in ['.unitigs.fa.gz', '.unitigs.fasta.gz', '.unitigs.fa', '.unitigs.fasta']:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
                unitig_files.add(name)
        log_report(f"\n📁 UNITIG FILES (data/unitigs/)")
        log_report("-" * 80)
        log_report(f"  Available unitig files: {len(unitig_files)}")
        
        # Check train coverage
        train_accessions = set(train_ids)
        train_available = train_accessions & unitig_files
        train_missing = train_accessions - unitig_files
        log_report(f"\n  TRAIN SET:")
        log_report(f"    Required:  {len(train_accessions)}")
        log_report(f"    Available: {len(train_available)} ({100*len(train_available)/len(train_accessions):.1f}%)")
        if len(train_missing) > 0:
            log_report(f"    Missing:   {len(train_missing)}")
        
        # Check test coverage
        test_accessions = set(test_ids)
        test_available = test_accessions & unitig_files
        test_missing = test_accessions - unitig_files
        log_report(f"\n  TEST SET:")
        log_report(f"    Required:  {len(test_accessions)}")
        log_report(f"    Available: {len(test_available)} ({100*len(test_available)/len(test_accessions):.1f}%)")
        if len(test_missing) > 0:
            log_report(f"    Missing:   {len(test_missing)}")
    else:
        log_report(f"\n📁 UNITIG FILES (data/unitigs/)")
        log_report("-" * 80)
        log_report(f"  Directory not found: {unitig_dir}")
    
    # Check validation FASTQ files
    fastq_dir = Path("data/validation/raw")
    if fastq_dir.exists():
        fastq_files = set()
        # Check for FASTQ files in subdirectories (one directory per accession)
        for subdir in fastq_dir.iterdir():
            if subdir.is_dir():
                # Directory name is the accession
                # Check if it contains fastq files
                has_fastq = any(subdir.glob('*.fastq.gz')) or any(subdir.glob('*.fq.gz')) or \
                           any(subdir.glob('*.fastq')) or any(subdir.glob('*.fq'))
                if has_fastq:
                    fastq_files.add(subdir.name)
        
        # Also check for FASTQ files directly in the raw directory (flat structure)
        for ext in ['*.fastq.gz', '*.fq.gz', '*.fastq', '*.fq']:
            for f in fastq_dir.glob(ext):
                # Extract accession from filename (handle _1.fastq.gz, _2.fastq.gz, etc.)
                name = f.name
                for suffix in ['_1.fastq.gz', '_2.fastq.gz', '_1.fq.gz', '_2.fq.gz', 
                              '.fastq.gz', '.fq.gz', '_1.fastq', '_2.fastq', '.fastq', '.fq']:
                    if name.endswith(suffix):
                        name = name[:-len(suffix)]
                        break
                fastq_files.add(name)
        
        log_report(f"\n📁 VALIDATION FASTQ FILES (data/validation/raw/)")
        log_report("-" * 80)
        log_report(f"  Available FASTQ accessions: {len(fastq_files)}")
        
        # Check validation coverage
        val_accessions = set(val_ids)
        val_available = val_accessions & fastq_files
        val_missing = val_accessions - fastq_files
        log_report(f"\n  VALIDATION SET:")
        log_report(f"    Required:  {len(val_accessions)}")
        log_report(f"    Available: {len(val_available)} ({100*len(val_available)/len(val_accessions):.1f}%)")
        if len(val_missing) > 0:
            log_report(f"    Missing:   {len(val_missing)}")
    else:
        log_report(f"\n📁 VALIDATION FASTQ FILES (data/validation/raw/)")
        log_report("-" * 80)
        log_report(f"  Directory not found: {fastq_dir}")
    
    # Summary
    log_report(f"\n" + "=" * 80)
    log_report("DOWNLOAD REQUIREMENTS")
    log_report("=" * 80)
    if unitig_dir.exists():
        total_unitig_needed = len(train_accessions | test_accessions)
        total_unitig_available = len((train_accessions | test_accessions) & unitig_files)
        total_unitig_missing = total_unitig_needed - total_unitig_available
        log_report(f"  Unitig files to download:  {total_unitig_missing} / {total_unitig_needed}")
    if fastq_dir.exists():
        log_report(f"  FASTQ files to download:   {len(val_missing)} / {len(val_accessions)}")
    
    # Save report
    report_path = output_dir / "label_distribution_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"Saved label distribution report to {report_path}")

    # Save config
    config = {
        "version": "v7",
        "method": "bioproject_disjoint_with_selective_label_coverage",
        "description": "BioProjects disjoint across splits. Forces community_type rare labels to training; accepts material rare labels as zero-shot. Stratifies host and env separately. Removes singleton classes.",
        "parameters": {
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "min_samples_per_class": args.min_samples_per_class,
            "quota_multiplier": 1.2,
        },
        "results": {
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "n_total": len(df),
            "n_train_bioprojects": len(train_df[bioproject_col].unique()),
            "n_val_bioprojects": len(val_df[bioproject_col].unique()),
            "n_test_bioprojects": len(test_df[bioproject_col].unique()),
        },
    }
    
    with open(output_dir / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n✨ Split v7 created successfully in {output_dir}/")


if __name__ == "__main__":
    main()
