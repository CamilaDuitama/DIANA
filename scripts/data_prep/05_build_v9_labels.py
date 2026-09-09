#!/usr/bin/env python3
"""Build the v9 label table from AncientMetagenomeDir v26.03.0.

Question answered
-----------------
v7's targets cap macro-F1 for reasons that have nothing to do with the model
(see results/label_space_audit/). v9 rebuilds the label space so the targets are
learnable and honestly scored. The runs and the feature matrix are unchanged —
only the labels.

What changes from v7
--------------------
1. `community_type` is split back into two targets. AMD defines
   `community_type` (host-associated, 7-term enum) and `feature` (environmental,
   14-term enum) as separate controlled vocabularies with an empty intersection;
   v7 concatenated them into one 17-class target that AMD's schema does not
   sanction. Each run now carries only the one that applies; the other is absent
   and is masked out of that head's loss and metrics.

2. Library rows are joined to samples on the sample ACCESSION, not `sample_name`
   (see 04_build_v7_metadata.py for why: `sample_name` is not unique, and the
   name-join let input row order decide 178 runs' labels).

3. Rows for the same run are MERGED rather than deduplicated. With
   `community_type` and `feature` separated, a run listed in both AMD tables --
   Rampelli2021's cave sediment carrying Neanderthal gut signal -- has two
   *complementary* rows, and merging keeps every correct label instead of
   discarding one. Only a target with two conflicting non-null values is masked,
   which leaves the 12 McDonough2018 runs where one SRA accession covers four
   tissues.

4. `sample_host` no longer encodes "no host" as a class. Environmental runs are
   masked for that head. In v7 they supplied free correct predictions and lifted
   apparent accuracy from 0.173 (real hosts) to 0.403.

5. `material` classes below a support threshold are masked. Support is counted on
   the TRAINING split only: pooling across splits would retain `rib` (132 runs)
   and `shallow marine sediment` (36), which have zero training runs and are
   unlearnable, and would let val/test define the label space -- the leakage the
   referees raised as R3.4.

Absent labels are written as empty cells and become IGNORE_INDEX at encoding
time, so they contribute no gradient and are excluded from metrics.

NOTE ON NAMING
--------------
This produces the *label table* only -- corrected targets, before any
partitioning. It is an intermediate consumed by
06_create_bioproject_splits_v9.py, which adds the BioProject-disjoint split and
writes data/splits_v9/. There is no v8 model, no v8 config and no v8 training
run; the label work and the partition work simply landed on different days.
Everything trained from here is v9.

Inputs
------
data/metadata/AncientMetagenomeDir-v26.03.0/*.tsv
data/splits_v7/{train,test,val}_accessions.txt   (same runs, same BioProject split)

Outputs
-------
data/v9_labels_prepartition/{train,test,val}_metadata.tsv
data/v9_labels_prepartition/label_space_report.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HOST_SAMPLE_COLS = ["community_type", "sample_host", "material",
                    "sample_age", "latitude", "longitude", "geo_loc_name"]
ENV_SAMPLE_COLS = ["feature", "material",
                   "sample_age", "latitude", "longitude", "geo_loc_name"]

# Targets that must agree across a run's library rows; a disagreement is masked.
TARGETS = ["community_type", "feature", "sample_host", "material",
           "sample_age", "latitude", "longitude"]

KEEP_COLS = [
    "Run_accession", "sample_type", "community_type", "feature", "sample_host",
    "material", "sample_age", "latitude", "longitude", "project_name",
    "publication_year", "geo_loc_name", "sample_name", "archive_project",
    "archive_sample_accession",
]


def join_libraries_to_samples(lib: pd.DataFrame, samples: pd.DataFrame,
                              sample_cols: list) -> pd.DataFrame:
    """Attach sample labels via the sample accession, not the name.

    `archive_accession` may hold a comma-separated list, so it is exploded first;
    without that, 305 library rows fail to match and lose their labels.
    """
    keyed = samples[["project_name", "archive_accession"] + sample_cols].copy()
    keyed["_acc"] = keyed["archive_accession"].astype(str).str.split(",")
    keyed = keyed.explode("_acc")
    keyed["_acc"] = keyed["_acc"].str.strip()
    keyed = keyed.drop(columns="archive_accession")
    return lib.merge(keyed,
                     left_on=["project_name", "archive_sample_accession"],
                     right_on=["project_name", "_acc"],
                     how="left").drop(columns="_acc")


def merge_run_rows(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Collapse a run's library rows into one, merging complementary values.

    For each target: take the single distinct non-null value if there is one,
    otherwise leave it absent (masked). Complementary rows -- one supplying
    `community_type`, the other `feature` -- therefore both survive, while a
    genuine disagreement is masked for that target only.
    """
    conflicts: dict[str, list[str]] = {t: [] for t in TARGETS}

    def collapse(group: pd.DataFrame) -> pd.Series:
        out = {}
        for target in TARGETS:
            vals = group[target].dropna().unique()
            if len(vals) == 1:
                out[target] = vals[0]
            elif len(vals) == 0:
                out[target] = pd.NA
            else:
                out[target] = pd.NA          # conflicting -> mask this target only
                conflicts[target].append(group.name)
        # Descriptive columns: first non-null is fine, they are not targets.
        for col in df.columns:
            if col in TARGETS or col == "Run_accession":
                continue
            nn = group[col].dropna()
            out[col] = nn.iloc[0] if len(nn) else pd.NA
        return pd.Series(out)

    merged = df.groupby("Run_accession", sort=False).apply(collapse)
    merged.index.name = "Run_accession"
    merged = merged.reset_index()

    report.append("Conflicting targets masked after the row merge:")
    for target, runs in conflicts.items():
        if runs:
            projects = sorted(set(df[df.Run_accession.isin(runs)].project_name.dropna()))
            report.append(f"  {target:16s} {len(runs):4d} runs  ({', '.join(projects)})")
    if not any(conflicts.values()):
        report.append("  none")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amd-dir", type=Path,
                    default=PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0")
    ap.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits_v7",
                    help="Source of the accession lists (v9 reuses the v7 BioProject split)")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/v9_labels_prepartition")
    ap.add_argument("--material-min-support", type=int, default=3,
                    help="Minimum TRAINING runs for a material class to be retained")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    A = args.amd_dir
    host_lib = pd.read_csv(A / "ancientmetagenome-hostassociated_libraries.tsv", sep="\t", low_memory=False)
    host_sam = pd.read_csv(A / "ancientmetagenome-hostassociated_samples.tsv", sep="\t", low_memory=False)
    env_lib = pd.read_csv(A / "ancientmetagenome-environmental_libraries.tsv", sep="\t", low_memory=False)
    env_sam = pd.read_csv(A / "ancientmetagenome-environmental_samples.tsv", sep="\t", low_memory=False)

    host_df = join_libraries_to_samples(host_lib, host_sam, HOST_SAMPLE_COLS)
    env_df = join_libraries_to_samples(env_lib, env_sam, ENV_SAMPLE_COLS)

    # The two vocabularies stay in separate columns. Environmental runs have no
    # host and no community_type; host-associated runs have no feature.
    host_df["feature"] = pd.NA
    env_df["community_type"] = pd.NA
    env_df["sample_host"] = pd.NA
    host_df["sample_type"] = "ancient_metagenome"
    env_df["sample_type"] = "ancient_metagenome"

    df = pd.concat([host_df, env_df], ignore_index=True)
    df = df.rename(columns={"archive_data_accession": "Run_accession"})
    report.append(f"Library rows merged: {len(df)}")

    df = merge_run_rows(df, report)
    report.append(f"Runs after merge: {len(df)}")

    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    # Split assignment (unchanged from v7 -- same BioProject-disjoint partition).
    splits = {}
    for name in ("train", "test", "val"):
        accs = set((args.splits_dir / f"{name}_accessions.txt").read_text().split())
        splits[name] = df[df.Run_accession.isin(accs)].copy()
        missing = accs - set(splits[name].Run_accession)
        if missing:
            report.append(f"WARNING: {len(missing)} {name} accessions absent from AMD")

    # material support on the TRAINING split only.
    support = splits["train"]["material"].value_counts()
    retained = set(support[support >= args.material_min_support].index)
    dropped = sorted(set(splits["train"]["material"].dropna()) - retained)
    report.append(f"\nmaterial: support counted on train only, min={args.material_min_support}")
    report.append(f"  retained {len(retained)} of {support.size} training classes")
    if dropped:
        report.append("  masked (too few training runs): "
                      + ", ".join(f"{c} ({int(support[c])})" for c in dropped))
    for name, sdf in splits.items():
        absent = sdf["material"].notna() & ~sdf["material"].isin(retained)
        report.append(f"  {name}: masked {int(absent.sum())} of {len(sdf)} runs "
                      f"({100 * absent.sum() / len(sdf):.1f}%)")
        sdf.loc[absent, "material"] = pd.NA

    # Report the resulting label space.
    report.append("\nLabel space per target (non-null runs / classes):")
    for target in ["community_type", "feature", "sample_host", "material"]:
        row = f"  {target:16s}"
        for name, sdf in splits.items():
            n = int(sdf[target].notna().sum())
            k = int(sdf[target].nunique(dropna=True))
            row += f"  {name}={n}/{k}cls"
        report.append(row)
    for target in ["sample_age", "latitude", "longitude"]:
        row = f"  {target:16s}"
        for name, sdf in splits.items():
            row += f"  {name}={int(sdf[target].notna().sum())}"
        report.append(row)

    for name, sdf in splits.items():
        out = args.output_dir / f"{name}_metadata.tsv"
        sdf.to_csv(out, sep="\t", index=False)
        report.append(f"\nWrote {out.name}: {len(sdf)} runs")

    text = "\n".join(report)
    print(text)
    (args.output_dir / "label_space_report.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
