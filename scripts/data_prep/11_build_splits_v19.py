#!/usr/bin/env python3
"""Build data/splits_v19/: the v17 BioProject split, unchanged, with v19's label rule.

PROTOCOLS.md v19, Camila's decisions 2026-10-09:
(1) the BioProject split stands; biosamples with runs on both sides are reported,
    not moved (data/splits_v19/biosample_overlap_train_test.tsv);
(3) material "unknown" means no material label: those runs become unlabelled for
    material (11 training runs, one BioProject, never evaluable).

Asserted: same runs, same sides and same dev folds as v17; no biosample spans two dev
folds; the eligibility table equals v17's apart from the removed "unknown" row; every
evaluable class has runs on both sides.
"""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/splits_v17"
DST = ROOT / "data/splits_v19"
GROUP = "archive_project"
SAMPLE = "archive_sample_accession"
UNKNOWN = ("material", "unknown")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _v17_builder():
    spec = importlib.util.spec_from_file_location("splits_v17", ROOT / "scripts/data_prep/10_build_splits_v17.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    DST.mkdir(exist_ok=False)
    tr = pd.read_csv(SRC / "train_metadata.tsv", sep="\t", dtype=str).fillna("")
    te = pd.read_csv(SRC / "test_metadata.tsv", sep="\t", dtype=str).fillna("")
    dev = pd.read_csv(SRC / "dev_folds.tsv", sep="\t")
    field, value = UNKNOWN
    n_tr, n_te = int((tr[field] == value).sum()), int((te[field] == value).sum())
    assert (n_tr, n_te) == (11, 0), (n_tr, n_te)
    tr.loc[tr[field] == value, field] = ""
    te.loc[te[field] == value, field] = ""
    log.info("material 'unknown' set to missing: %d training runs, %d held-out runs", n_tr, n_te)

    assert list(dev.Run_accession) == list(tr.Run_accession), "dev folds must list the training runs in order"
    fold = dev.set_index("Run_accession").fold
    spans = tr.assign(fold=tr.Run_accession.map(fold)).groupby(SAMPLE).fold.nunique()
    assert (spans <= 1).all(), f"{int((spans > 1).sum())} biosamples span two dev folds"

    shared = sorted(set(tr[SAMPLE]) & set(te[SAMPLE]))
    overlap = pd.concat([tr[tr[SAMPLE].isin(shared)].assign(side="train"),
                         te[te[SAMPLE].isin(shared)].assign(side="held-out")])
    overlap = overlap[["side", SAMPLE, "Run_accession", GROUP, "project_name",
                       "community_type", "feature", "sample_host", "material"]]
    log.info("biosamples with runs on both sides (kept, decision 1): %d, runs %d train / %d held-out",
             len(shared), int((overlap.side == "train").sum()), int((overlap.side == "held-out").sum()))

    elig = _v17_builder().class_eligibility(pd.concat([tr, te], ignore_index=True))
    old = pd.read_csv(SRC / "class_eligibility.tsv", sep="\t")
    old = old[~((old.target == field) & (old["class"] == value))]
    key = ["target", "class", "n_runs", "n_bioprojects", "evaluable"]
    a = elig[key].sort_values(key[:2]).reset_index(drop=True)
    b = old[key].sort_values(key[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    for _, r in elig[elig.evaluable].iterrows():
        assert (tr[r.target] == r["class"]).any() and (te[r.target] == r["class"]).any(), r.to_dict()

    tr.to_csv(DST / "train_metadata.tsv", sep="\t", index=False)
    te.to_csv(DST / "test_metadata.tsv", sep="\t", index=False)
    tr.Run_accession.to_csv(DST / "train_accessions.txt", index=False, header=False)
    te.Run_accession.to_csv(DST / "test_accessions.txt", index=False, header=False)
    dev.to_csv(DST / "dev_folds.tsv", sep="\t", index=False)
    elig.to_csv(DST / "class_eligibility.tsv", sep="\t", index=False)
    overlap.to_csv(DST / "biosample_overlap_train_test.tsv", sep="\t", index=False)
    json.dump({"version": "v19", "split": "data/splits_v17, unchanged (same runs, sides, dev folds)",
               "label_rule": "material 'unknown' -> missing (11 training runs)",
               "biosamples_on_both_sides": len(shared),
               "protocol": "PROTOCOLS.md v19, decisions (1) and (3)", "built": "2026-10-09"},
              open(DST / "split_config.json", "w"), indent=1)
    log.info("wrote %s", DST)


if __name__ == "__main__":
    main()
