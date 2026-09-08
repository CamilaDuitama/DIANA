#!/usr/bin/env python
"""
Audit the v7 split labels against an AncientMetagenomeDir release.

Re-derives labels straight from the AMD TSVs using the same join as
scripts/data_prep/04_build_v7_metadata.py, then compares them to what is
actually in data/splits_v7/{train,val,test}_metadata.tsv. Reports, per label
column: disagreements, values present in the split but absent from AMD, missing
rates, and where each community_type value came from (host-associated
`community_type` vs environmental `feature`).

Usage:
    ./env/bin/python scripts/analysis/audit_v7_labels.py [--amd-dir DIR]

Output: results/label_audit/ (mismatch tables + summary.txt)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["community_type", "sample_host", "material", "sample_age",
          "latitude", "longitude"]


def find_tsv(amd_dir: Path, stem: str) -> Path:
    """Locate an AMD table by filename stem.

    Works for both a flat snapshot directory and a raw `git clone` of
    spaam-community/AncientMetagenomeDir, which nests the tables under
    ancientmetagenome-<kind>/.
    """
    direct = amd_dir / f"{stem}.tsv"
    if direct.exists():
        return direct
    hits = sorted(amd_dir.rglob(f"{stem}.tsv"))
    if not hits:
        sys.exit(f"Could not find {stem}.tsv under {amd_dir}")
    return hits[0]


def load_amd(amd_dir: Path) -> pd.DataFrame:
    """Reproduce the 04_build_v7_metadata.py join, keeping provenance."""
    hl = pd.read_csv(find_tsv(amd_dir, "ancientmetagenome-hostassociated_libraries"),
                     sep="\t", low_memory=False)
    hs = pd.read_csv(find_tsv(amd_dir, "ancientmetagenome-hostassociated_samples"),
                     sep="\t", low_memory=False)
    el = pd.read_csv(find_tsv(amd_dir, "ancientmetagenome-environmental_libraries"),
                     sep="\t", low_memory=False)
    es = pd.read_csv(find_tsv(amd_dir, "ancientmetagenome-environmental_samples"),
                     sep="\t", low_memory=False)

    host = hl.merge(
        hs[["project_name", "sample_name", "community_type", "sample_host",
            "material", "sample_age", "latitude", "longitude", "geo_loc_name"]],
        on=["project_name", "sample_name"], how="left")
    host["source_table"] = "hostassociated"

    env = el.merge(
        es[["project_name", "sample_name", "feature", "material",
            "sample_age", "latitude", "longitude", "geo_loc_name"]],
        on=["project_name", "sample_name"], how="left")
    env["community_type"] = env["feature"]
    env["sample_host"] = None
    env["source_table"] = "environmental"

    df = pd.concat([host, env], ignore_index=True)
    df = df.rename(columns={"archive_data_accession": "Run_accession"})

    n_rows = len(df)
    dup = df["Run_accession"].duplicated(keep=False).sum()
    df["_completeness"] = df[["community_type", "sample_host",
                              "material"]].notna().sum(axis=1)
    df = (df.sort_values("_completeness", ascending=False)
            .drop_duplicates(subset="Run_accession", keep="first")
            .drop(columns="_completeness"))
    return df, n_rows, dup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amd-dir", type=Path,
                    default=ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits_v7")
    ap.add_argument("--out", type=Path, default=ROOT / "results/label_audit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.amd_dir.exists():
        sys.exit(f"AMD dir not found: {args.amd_dir}")

    amd, n_lib_rows, n_dup = load_amd(args.amd_dir)
    lines = [f"AMD release dir : {args.amd_dir.name}",
             f"library rows    : {n_lib_rows}",
             f"rows sharing a Run_accession with another row : {n_dup}",
             f"unique Run_accession after dedup : {len(amd)}",
             f"Run_accession values containing a separator (',' or ';') : "
             f"{amd['Run_accession'].astype(str).str.contains('[,;]').sum()}",
             ""]

    splits = {}
    for name in ["train", "val", "test"]:
        p = args.splits_dir / f"{name}_metadata.tsv"
        if p.exists():
            splits[name] = pd.read_csv(p, sep="\t", low_memory=False)

    all_mismatch = []
    for name, sp in splits.items():
        m = sp.merge(amd[["Run_accession", "source_table"] + LABELS],
                     on="Run_accession", how="left", suffixes=("_split", "_amd"))
        not_in_amd = m["source_table"].isna().sum()
        lines.append(f"[{name}] n={len(sp)}  not found in AMD: {not_in_amd}")
        lines.append(f"[{name}] source_table: "
                     + m["source_table"].value_counts(dropna=False).to_dict().__str__())

        for col in LABELS:
            a, b = m[f"{col}_split"], m[f"{col}_amd"]
            if col in ("sample_age", "latitude", "longitude"):
                a_n, b_n = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
                diff = ~((a_n - b_n).abs() < 1e-6) & ~(a_n.isna() & b_n.isna())
            else:
                a_s = a.astype(str).str.strip().str.lower()
                b_s = b.astype(str).str.strip().str.lower()
                diff = (a_s != b_s) & ~(a.isna() & b.isna())
            diff = diff & m["source_table"].notna()
            lines.append(f"[{name}] {col:15s} disagreements={int(diff.sum()):4d}"
                         f"  missing_in_split={int(a.isna().sum()):4d}"
                         f"  missing_in_amd={int(b.isna().sum()):4d}")
            if diff.any():
                sub = m.loc[diff, ["Run_accession", "project_name", "source_table",
                                   f"{col}_split", f"{col}_amd"]].copy()
                sub.insert(0, "label_column", col)
                sub.insert(0, "split", name)
                all_mismatch.append(sub.rename(
                    columns={f"{col}_split": "value_in_split",
                             f"{col}_amd": "value_in_amd"}))
        lines.append("")

    if all_mismatch:
        mm = pd.concat(all_mismatch, ignore_index=True)
        mm.to_csv(args.out / "label_mismatches.csv", index=False)
        lines.append(f"Wrote {len(mm)} mismatch rows to label_mismatches.csv")
    else:
        lines.append("No label disagreements found.")
    lines.append("")

    # Where does each community_type value come from?
    full = pd.concat(splits.values(), ignore_index=True)
    prov = full.merge(amd[["Run_accession", "source_table"]],
                      on="Run_accession", how="left")
    ct = (prov.groupby(["community_type", "source_table"], dropna=False)
              .size().unstack(fill_value=0))
    ct.to_csv(args.out / "community_type_provenance.csv")
    lines.append("community_type by AMD source table "
                 "(environmental values come from the `feature` column):")
    lines.append(ct.to_string())
    lines.append("")

    for col in ["community_type", "sample_host", "material"]:
        vc = full[col].value_counts(dropna=False)
        singles = vc[vc <= 2]
        lines.append(f"{col}: {full[col].nunique()} distinct values, "
                     f"{int(full[col].isna().sum())} missing, "
                     f"{len(singles)} values with <=2 runs across all splits")
    txt = "\n".join(lines)
    (args.out / "summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
