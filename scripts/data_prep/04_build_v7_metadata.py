#!/usr/bin/env python3
"""
Build train/test/val metadata TSVs for the v7 split from AncientMetagenomeDir v26.03.0.

Produces files with the same format as data/splits_v5/train_metadata.tsv so that
downstream scripts (baselines, evaluation) work without modification.

Required columns (minimum for baselines and model training):
    Run_accession, sample_type, community_type, sample_host, material,
    sample_age, latitude, longitude, project_name, publication_year, geo_loc_name

Usage:
    python scripts/data_prep/04_build_v7_metadata.py
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Paths ──────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--amd-dir", type=Path,
                default=PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0")
_p.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits_v7",
                help="Directory holding {train,test,val}_accessions.txt")
_p.add_argument("--output-dir", type=Path, default=None,
                help="Where to write the metadata TSVs (default: --splits-dir, i.e. in place)")
_args = _p.parse_args()

AMD_DIR      = _args.amd_dir
SPLITS_DIR   = _args.splits_dir
OUTPUT_DIR   = _args.output_dir or _args.splits_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOST_LIB     = AMD_DIR / "ancientmetagenome-hostassociated_libraries.tsv"
HOST_SAMPLES = AMD_DIR / "ancientmetagenome-hostassociated_samples.tsv"
ENV_LIB      = AMD_DIR / "ancientmetagenome-environmental_libraries.tsv"
ENV_SAMPLES  = AMD_DIR / "ancientmetagenome-environmental_samples.tsv"

TRAIN_ACC    = SPLITS_DIR / "train_accessions.txt"
TEST_ACC     = SPLITS_DIR / "test_accessions.txt"
VAL_ACC      = SPLITS_DIR / "val_accessions.txt"

# ── Load & merge metadata (same logic as 03_create_bioproject_splits_v7_fixed) ─
print("Loading AMD metadata...")
host_lib     = pd.read_csv(HOST_LIB,     sep="\t", low_memory=False)
host_samples = pd.read_csv(HOST_SAMPLES, sep="\t", low_memory=False)
env_lib      = pd.read_csv(ENV_LIB,      sep="\t", low_memory=False)
env_samples  = pd.read_csv(ENV_SAMPLES,  sep="\t", low_memory=False)

def join_libraries_to_samples(lib: pd.DataFrame, samples: pd.DataFrame,
                              sample_cols: list) -> pd.DataFrame:
    """Attach sample labels to library rows via the sample ACCESSION, not the name.

    `sample_name` is not unique within a project — 73 host-associated and 8
    environmental (project_name, sample_name) pairs name more than one physical
    specimen. Joining on it fans one library row out to several sample rows with
    conflicting labels, and the completeness tie-break below then picks one
    essentially at random: 178 host-associated runs took a different label purely
    from input row order.

    The library row already records which specimen it came from, in
    `archive_sample_accession`. The samples table's `archive_accession` may hold a
    comma-separated list of accessions (e.g. "ERS2985755,ERS3881523"), so it is
    split and exploded before joining — without that, 305 library rows fail to
    match and would lose their labels.

    This takes host-associated ambiguity from 178 runs to 12 with no loss of
    coverage (23 unlabelled runs either way). The residual 12 are an upstream AMD
    collision in McDonough2018, where one SRA sample accession covers four
    different tissues of the same museum specimen.
    """
    keyed = samples[['project_name', 'archive_accession'] + sample_cols].copy()
    keyed['_acc'] = keyed['archive_accession'].astype(str).str.split(',')
    keyed = keyed.explode('_acc')
    keyed['_acc'] = keyed['_acc'].str.strip()
    keyed = keyed.drop(columns='archive_accession')

    return lib.merge(
        keyed,
        left_on=['project_name', 'archive_sample_accession'],
        right_on=['project_name', '_acc'],
        how='left',
    ).drop(columns='_acc')


host_df = join_libraries_to_samples(
    host_lib, host_samples,
    ['community_type', 'sample_host', 'material', 'sample_age',
     'latitude', 'longitude', 'geo_loc_name'])
env_df = join_libraries_to_samples(
    env_lib, env_samples,
    ['feature', 'material', 'sample_age', 'latitude', 'longitude', 'geo_loc_name'])

host_df['sample_type']    = 'ancient_metagenome'
env_df['sample_type']     = 'ancient_metagenome'
env_df['community_type']  = env_df['feature']
env_df['sample_host']     = None

df = pd.concat([host_df, env_df], ignore_index=True)

# Rename accession column to match v5 convention
df = df.rename(columns={'archive_data_accession': 'Run_accession'})

print(f"Total merged libraries before dedup: {len(df)}")

# Deduplicate: same run may appear in multiple library rows (different preps).
# Keep the row with the most non-null label columns per Run_accession.
label_cols = ['community_type', 'sample_host', 'material', 'sample_type']
df['_label_completeness'] = df[label_cols].notna().sum(axis=1)

# Report what the tie-break still has to decide. After the accession join this
# should be the 12 McDonough2018 runs only; anything more means the join regressed.
_conflicts = (df.groupby('Run_accession')[['community_type', 'material']]
                .nunique().gt(1).any(axis=1))
if _conflicts.any():
    print(f"NOTE: {int(_conflicts.sum())} runs still have conflicting labels "
          f"after the accession join (expected: 12, all McDonough2018):")
    for run in _conflicts[_conflicts].index[:15]:
        print(f"  {run}")

# mergesort is stable and the secondary keys are explicit, so the surviving row is
# fully determined by the data rather than by input row order.
df = (df.sort_values(['_label_completeness', 'Run_accession', 'sample_name'],
                     ascending=[False, True, True], kind='mergesort')
        .drop_duplicates(subset='Run_accession', keep='first')
        .drop(columns='_label_completeness'))
print(f"Total after dedup: {len(df)}")

# ── Select output columns ──────────────────────────────────────────────────────
KEEP_COLS = [
    'Run_accession', 'sample_type', 'community_type', 'sample_host', 'material',
    'sample_age', 'latitude', 'longitude', 'project_name', 'publication_year',
    'geo_loc_name', 'sample_name', 'archive_project', 'archive_sample_accession',
]
# Keep only columns that exist
KEEP_COLS = [c for c in KEEP_COLS if c in df.columns]
df = df[KEEP_COLS].copy()

# ── Write split TSVs ───────────────────────────────────────────────────────────
for split_name, acc_file in [('train', TRAIN_ACC), ('test', TEST_ACC), ('val', VAL_ACC)]:
    accessions = set(acc_file.read_text().splitlines())
    split_df   = df[df['Run_accession'].isin(accessions)].copy()

    # Warn about missing accessions
    missing = accessions - set(split_df['Run_accession'])
    if missing:
        print(f"WARNING: {len(missing)} {split_name} accessions not found in AMD metadata:")
        for a in sorted(missing)[:10]:
            print(f"  {a}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    out_path = OUTPUT_DIR / f"{split_name}_metadata.tsv"
    split_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {split_name}_metadata.tsv: {len(split_df)} samples, "
          f"{split_df['community_type'].notna().sum()} with community_type, "
          f"{split_df['material'].notna().sum()} with material")

print("\nDone. Files written to", OUTPUT_DIR)
