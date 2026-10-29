#!/usr/bin/env python3
"""Faure et al. organism-tag corrections for DIANA's runs, beside the AncientMetagenomeDir labels and DIANA's predictions.

For each of our runs in Roland Faure's correction list (/pasteur/gaia/homes/cduitama/metadata_correction.tsv):
  - the SRA organism tag and their corrected tag;
  - the AncientMetagenomeDir (AMD) sample row of the run (libraries table -> sample), its host, community type and material
    (v26.03.0, the release the splits use), checked against our split metadata;
  - the AMD commit at which the sample first carried these three labels (earliest commit of the samples table whose row for
    the sample has them), with its GitHub link and the first AMD release (tag) that contains it;
  - DIANA's prediction for the three fields from a model that never saw the run (out-of-fold dev predictions for training runs,
    the released model for held-out runs), with the no-call cutoffs.
Writes results/faure_corrections_v19/faure_amd_diana.csv (one row per run) and prints a summary by BioProject.

    ./env/bin/python scripts/analysis/78_faure_amd_table_v19.py
"""
from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAURE = Path("/pasteur/gaia/homes/cduitama/metadata_correction.tsv")
AMD = ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"
REPO = ROOT / "data/metadata/AncientMetagenomeDir-upstream"
SAMPLES = "ancientmetagenome-hostassociated/samples/ancientmetagenome-hostassociated_samples.tsv"
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/faure_corrections_v19"
DEV_DIR, HELD_DIR = "results/devfold_probs_v19", "results/heldout_v19/diana_refit"
FIELDS = {"sample_host": "sample_host", "community_type": "community_type", "material": "material"}
URL = "https://github.com/SPAAM-community/AncientMetagenomeDir/commit/{}"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def labels_at(sha: str, path: str, project: str, sample: str) -> tuple | None:
    try:
        d = pd.read_csv(io.StringIO(git("show", f"{sha}:{path}")), sep="\t", dtype=str)
    except (subprocess.CalledProcessError, pd.errors.ParserError):
        return None
    if not {"project_name", "sample_name", *FIELDS} <= set(d.columns):
        return None
    r = d[(d.project_name == project) & (d.sample_name == sample)]
    return None if r.empty else tuple(r.iloc[0].get(c) for c in FIELDS)


def assigned(project: str, sample: str, labels: tuple) -> tuple[str, str]:
    """Earliest commit whose samples table (followed across renames) gives this sample these labels, and the first
    versioned release (vYY.MM...) containing it."""
    log = git("log", "--follow", "--name-only", "--format=@%H", "-S", sample, "--", SAMPLES).split("\n")
    commits, sha = [], None
    for line in log:
        if line.startswith("@"):
            sha = line[1:]
        elif line.strip() and sha:
            commits.append((sha, line.strip())); sha = None
    for sha, path in reversed(commits):
        if labels_at(sha, path, project, sample) == labels:
            tags = [t for t in git("tag", "--contains", sha, "--sort=v:refname").split() if t[:1] == "v" and t[1:3].isdigit() and int(t[1:3]) >= 20]
            return sha, (tags[0] if tags else "unreleased")
    return "", ""


def diana(run: str, split: str, cut: dict) -> dict:
    out = {}
    for f in FIELDS:
        path = (ROOT / f"{DEV_DIR}/probs_DIANA_{f}.tsv" if split == "training" else ROOT / f"{HELD_DIR}/probs_DIANA_{f}.tsv")
        p = pd.read_csv(path, sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
        if run not in p.index:
            out[f"diana_{f}"] = ""
            continue
        q = p.loc[run, [c for c in p.columns if c.startswith("p_")]].astype(float)
        out[f"diana_{f}"] = q.idxmax()[2:] if q.max() >= cut[f] else f"no call ({q.idxmax()[2:]})"
    return out


def main() -> None:
    global OUT, DEV_DIR, HELD_DIR
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    _p.add_argument("--faure", type=Path, default=FAURE, help="correction list (compute nodes: a copy under data/external/faure)")
    a = _p.parse_known_args()[0]
    if a.net:
        DEV_DIR, HELD_DIR, OUT = "results/devfold_probs_v19net", "results/heldout_v19/diana_net", ROOT / "results/faure_corrections_v19net"
    fc = pd.read_csv(a.faure, sep="\t", dtype=str).set_index("Accession")
    meta = pd.concat([pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).assign(split="training"),
                      pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).assign(split="held-out")]).set_index("Run_accession")
    lib = pd.read_csv(AMD / "ancientmetagenome-hostassociated_libraries.tsv", sep="\t", dtype=str).drop_duplicates("archive_data_accession")
    lib = lib.set_index("archive_data_accession")
    smp = pd.read_csv(AMD / "ancientmetagenome-hostassociated_samples.tsv", sep="\t", dtype=str)
    cut = json.load(open(ROOT / ("configs/abstention_v19net.json" if a.net else "configs/abstention_v19.json")))["min_confidence"]
    rows = []
    for run in fc.index.intersection(meta.index):
        m, f = meta.loc[run], fc.loc[run]
        row = {"run": run, "bioproject": f.BioProject, "split": m.split, "sra_tag": f["Organism tag"], "faure_correction": f["Corrected tag"]}
        if run in lib.index:
            L = lib.loc[run]
            s = smp[(smp.project_name == L.project_name) & (smp.sample_name == L.sample_name)]
            if s.empty:   # AMD's own join gives no sample row, so the run has no AMD label
                row.update({"amd_project": L.project_name, "amd_sample": L.sample_name, "our_labels_used": "none"})
                row.update(diana(run, m.split, cut))
                rows.append(row)
                continue
            if len(s) != 1:
                raise ValueError(f"{run}: {len(s)} AMD sample rows for {L.sample_name}")
            s = s.iloc[0]
            labels = tuple(s[c] for c in FIELDS)
            for c, ours in FIELDS.items():
                if not (pd.isna(m[ours]) or m[ours] == s[c]):
                    raise ValueError(f"{run}: our {ours} {m[ours]!r} differs from AMD {s[c]!r}")
            sha, tag = assigned(s.project_name, L.sample_name, labels)
            row.update({"amd_project": L.project_name, "amd_sample": L.sample_name, "amd_sample_project": s.project_name, "amd_host": s.sample_host,
                        "amd_community_type": s.community_type, "amd_material": s.material,
                        "our_labels_used": "; ".join(f"{c}={m[o]}" for c, o in FIELDS.items() if pd.notna(m[o])) or "none",
                        "amd_commit": sha[:10], "amd_commit_url": URL.format(sha) if sha else "", "first_amd_release": tag})
        else:
            row.update({"amd_project": "not in AMD libraries table"})
        row.update(diana(run, m.split, cut))
        rows.append(row)
    d = pd.DataFrame(rows).sort_values(["bioproject", "run"])
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "faure_amd_diana.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["bioproject", "amd_project", "amd_sample_project", "sra_tag", "faure_correction", "amd_host", "amd_community_type", "amd_material",
            "our_labels_used", "diana_sample_host", "diana_community_type", "diana_material", "amd_commit", "first_amd_release"]
    print(d.groupby(cols, dropna=False).size().rename("runs").reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
