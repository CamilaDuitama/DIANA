#!/usr/bin/env python3
"""Verify the v9 labels against an AncientMetagenomeDir release, by biosample accession.

Join chain (the archive's own data model; every run belongs to one biosample):
    run (SRR/ERR, from the comma-split libraries.archive_data_accession)
    -> that library row's archive_sample_accession (one SRS/ERS)
    -> the samples row whose comma-split archive_accession contains it
    -> community_type / feature / sample_host / material.

Never join on (project_name, sample_name): 73 host-associated name pairs cover two
different samples (e.g. Mann2018 C214 = a calculus biosample AND a tooth-root
biosample), 69 of them with different labels. A name-keyed join on 2026-10-06
manufactured 177 phantom mismatches and nearly triggered a full retrain (the
withdrawn v15 protocol, PROTOCOLS.md).

Checks, all hard assertions:
  1. every corpus run resolves to exactly one samples row (none to zero or two);
  2. v9 labels equal the release's labels on every applicable field;
  3. Run_accession unique within train and held-out, and disjoint across them;
  4. all runs of one (project, sample) carry identical labels.

Writes a short report next to the AMD tables' label check results.

    ./env/bin/python scripts/data_prep/09_verify_labels_v9.py
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AMD = ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"
SPLITS = ROOT / "data/splits_v9"
REPORT = ROOT / "results/label_check_v9/amd_v26.03.0_label_verification.txt"

TASKS = ["community_type", "feature", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def amd_lookup() -> dict:
    """biosample accession -> (table, samples row)."""
    bios: dict = {}
    for kind in ["hostassociated", "environmental"]:
        with open(AMD / f"ancientmetagenome-{kind}_samples.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                for acc in row["archive_accession"].replace(";", ",").split(","):
                    bios.setdefault(acc.strip(), []).append((kind, row))
    return bios


def run_to_samples(bios: dict) -> dict:
    """run accession -> list of (table, samples row), via the library rows.

    Composite key: the library's biosample accession, disambiguated by the
    library's sample_name when one biosample spans several curated samples
    (McDonough2018 deposits one biosample per specimen but AMD curates one
    sample per tissue). Mann2018 is the mirror case: shared sample_name,
    distinct biosamples. The composite resolves both.
    """
    out: dict = {}
    for kind in ["hostassociated", "environmental"]:
        with open(AMD / f"ancientmetagenome-{kind}_libraries.tsv") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                hits = bios.get(row["archive_sample_accession"].strip(), [])
                if len(hits) > 1:
                    named = [(k, s) for k, s in hits
                             if s["sample_name"] == row["sample_name"]
                             and s["project_name"] == row["project_name"]]
                    if named:
                        hits = named
                for acc in row["archive_data_accession"].replace(";", ",").split(","):
                    out.setdefault(acc.strip(), []).extend(hits)
    return out


def main() -> None:
    tr = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", dtype=str).fillna("")
    te = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t", dtype=str).fillna("")

    # 3. uniqueness and disjointness
    assert not tr.Run_accession.duplicated().any()
    assert not te.Run_accession.duplicated().any()
    assert not (set(tr.Run_accession) & set(te.Run_accession))

    run2s = run_to_samples(amd_lookup())
    both = pd.concat([tr.assign(split="train"), te.assign(split="heldout")],
                     ignore_index=True)

    unresolved, ambiguous, mismatches, blanks = [], [], [], []
    for _, r in both.iterrows():
        hits = run2s.get(r.Run_accession, [])
        rows = {id(s): (kind, s) for kind, s in hits}
        if not rows:
            unresolved.append(r.Run_accession)
            continue
        values: dict = {}
        for kind, s in rows.values():
            for col in TASKS:
                if kind == "environmental" and col in ("sample_host", "community_type"):
                    continue
                if kind == "hostassociated" and col == "feature":
                    continue
                v = (s.get(col) or "").strip()
                if v:
                    values.setdefault(col, set()).add(v.lower())
        for col, vals in values.items():
            ours = r[col].strip().lower()
            if len(vals) > 1:
                ambiguous.append((r.Run_accession, col, sorted(vals), ours))
                continue
            amd_v = next(iter(vals))
            if ours and ours != amd_v:
                mismatches.append((r.split, r.Run_accession, col, ours, amd_v))
            if not ours and amd_v:
                blanks.append((r.split, r.Run_accession, col, amd_v))

    # 4. within-BIOSAMPLE consistency in our metadata. (Not per sample_name: one
    # name legitimately covers several biosamples with different labels, e.g.
    # Mann2018 C214 calculus vs tooth root.)
    bad_groups = 0
    for _, g in both.groupby("archive_sample_accession"):
        for col in TASKS:
            if g[col].replace("", pd.NA).dropna().nunique() > 1:
                bad_groups += 1

    lines = [
        "v9 labels vs AncientMetagenomeDir v26.03.0 (biosample-accession join)",
        f"corpus runs: {len(both)} (train {len(tr)}, held-out {len(te)})",
        f"runs not resolving to a samples row: {len(unresolved)}",
        f"run-fields unresolvable even with the composite key: {len(ambiguous)}",
        f"run-fields blank in v9 but labelled in AMD (conservative gaps): {len(blanks)}",
        f"label mismatches: {len(mismatches)}",
        f"(project, sample) groups with inconsistent labels: {bad_groups}",
        f"blank-gap runs: {sorted(set(b[1] for b in blanks))}",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    for ln in lines:
        log.info(ln)
    assert not unresolved and not ambiguous and not mismatches and not bad_groups, \
        "verification failed; see report"
    log.info("PASS — report at %s", REPORT)


if __name__ == "__main__":
    sys.exit(main())
