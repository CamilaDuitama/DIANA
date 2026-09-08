#!/usr/bin/env python
"""How many real metadata corrections can we test an anomaly detector against?

Question answered
-----------------
The most convincing demonstration for R3.6 is not planted noise but *real*
mislabels: samples whose metadata AncientMetagenomeDir itself later corrected.
This walks the AMD git history commit by commit, finds every case where an
already-present sample's label changed, and reports how many are usable -- i.e.
how many correspond to runs that are in our splits and have features.

Change types are separated, because they are not equally useful:

  GENUINE relabel          oral -> skeletal tissue.       A real correction.
  vocabulary normalisation calculus -> dental calculus.   Same sample, tidier term.
  fill/blank               NaN -> sediment.               Metadata added, not fixed.

Only the first is ground truth for "the metadata was wrong and a detector should
have flagged it".

Release tags are not enough: corrections are usually made and absorbed within a
release, so comparing v25.09 to v26.03 finds almost nothing while a commit-level
walk finds far more.

Inputs : data/metadata/AncientMetagenomeDir-upstream (full clone with history)
         data/splits_v9/*_accessions.txt
Outputs: results/amd_label_corrections/{corrections.tsv,summary.txt}
"""
from __future__ import annotations

import argparse
import io
import logging
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMD = PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-upstream"
TABLES = {"hostassociated": ("community_type", "sample_host", "material"),
          "environmental": ("feature", "material")}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(AMD)] + list(args),
                          capture_output=True, text=True).stdout


def classify(old: str, new: str) -> str:
    o, n = str(old).strip().lower(), str(new).strip().lower()
    if o in ("nan", "none", "") or n in ("nan", "none", ""):
        return "fill/blank"
    if o.replace(" ", "") in n.replace(" ", "") or n.replace(" ", "") in o.replace(" ", ""):
        return "vocabulary normalisation"
    return "genuine relabel"


def mine() -> pd.DataFrame:
    rows = []
    for kind, fields in TABLES.items():
        rel = f"ancientmetagenome-{kind}/samples/ancientmetagenome-{kind}_samples.tsv"
        commits = git("log", "--format=%H", "--reverse", "--", rel).split()
        logger.info("%s: walking %d commits", kind, len(commits))
        prev = None
        for sha in commits:
            txt = git("show", f"{sha}:{rel}")
            if not txt.strip():
                continue
            try:
                cur = pd.read_csv(io.StringIO(txt), sep="\t", low_memory=False)
                if "archive_accession" not in cur:
                    continue
                cur = cur.drop_duplicates("archive_accession").set_index("archive_accession")
            except Exception:
                continue
            if prev is not None:
                shared = prev.index.intersection(cur.index)
                for f in fields:
                    if f in prev and f in cur:
                        changed = prev.loc[shared, f].astype(str) != cur.loc[shared, f].astype(str)
                        for acc in shared[changed]:
                            rows.append({"table": kind, "sample_accession": acc, "field": f,
                                         "old": str(prev.loc[acc, f]),
                                         "new": str(cur.loc[acc, f]), "commit": sha[:8]})
            prev = cur
    df = pd.DataFrame(rows)
    df["change_type"] = [classify(o, n) for o, n in zip(df.old, df.new)]
    return df


def map_to_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Sample accessions are not run accessions; our splits are keyed on runs."""
    lib = []
    for kind in TABLES:
        rel = f"ancientmetagenome-{kind}/libraries/ancientmetagenome-{kind}_libraries.tsv"
        txt = git("show", f"HEAD:{rel}")
        d = pd.read_csv(io.StringIO(txt), sep="\t", low_memory=False)
        lib.append(d[["archive_sample_accession", "archive_data_accession"]])
    lib = pd.concat(lib, ignore_index=True).dropna()
    return df.merge(lib, left_on="sample_accession",
                    right_on="archive_sample_accession", how="left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits_v9")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/amd_label_corrections")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not AMD.exists():
        logger.error("AMD clone not found at %s", AMD)
        return 1

    df = map_to_runs(mine())
    splits = {s: set((args.splits / f"{s}_accessions.txt").read_text().split())
              for s in ("train", "val", "test")}
    df["split"] = [next((s for s, a in splits.items() if r in a), "not_in_v9")
                   for r in df.archive_data_accession.fillna("")]

    lines = ["AMD historical label corrections", ""]
    lines.append(f"field-level changes found : {len(df)}")
    lines.append(f"distinct samples          : {df.sample_accession.nunique()}")
    lines.append(f"distinct runs             : {df.archive_data_accession.nunique()}")
    lines.append("")
    lines.append("by change type:")
    for k, v in df.change_type.value_counts().items():
        lines.append(f"    {k:26s} {v:5d}")
    lines.append("")
    g = df[df.change_type == "genuine relabel"]
    lines.append(f"GENUINE relabels: {len(g)} changes over {g.archive_data_accession.nunique()} runs")
    lines.append("  by table/field:")
    for (tb, f), n in g.groupby(["table", "field"]).size().items():
        lines.append(f"    {tb:16s} {f:16s} {n:5d}")
    lines.append("")
    lines.append("  usable now — genuine relabels whose run is in a v9 split:")
    for s, n in g.split.value_counts().items():
        lines.append(f"    {s:12s} {n:5d}")
    lines.append("")
    lines.append("  most common corrections:")
    for k, v in (g.old + "  ->  " + g.new).value_counts().head(10).items():
        lines.append(f"    {v:4d}x  {k}")

    report = "\n".join(lines)
    print(report)
    df.to_csv(args.output / "corrections.tsv", sep="\t", index=False)
    (args.output / "summary.txt").write_text(report + "\n")
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
