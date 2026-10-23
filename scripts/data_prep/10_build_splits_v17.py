#!/usr/bin/env python3
"""Build data/splits_v17/: the optimal BioProject partition (PROTOCOLS.md v17).

Inputs are the v9 metadata (labels verified identical to AMD v26.03.0) and the
two frozen assignments produced by the integer program on 2026-10-07:
results/split_optim_v17/optimal_mean_heldout_projects.txt (held-out projects)
and optimal_devfolds_assignment.tsv (training project -> dev fold).

Everything the protocol promises is asserted here: disjointness, the exact
run/project counts, every >=2-project eligible class present on both sides,
every >=2-training-project class spanning >= 2 dev folds, fold sizes 17-23 %.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/splits_v9"
OPT = ROOT / "results/split_optim_v17"
DST = ROOT / "data/splits_v17"

TASKS = ["community_type", "feature", "sample_host", "material"]
GROUP = "archive_project"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def class_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TASKS:
        sub = df[df[t].notna() & (df[t] != "")]
        for cls, g in sub.groupby(t):
            rows.append({"target": t, "class": cls, "n_runs": len(g),
                         "n_bioprojects": g[GROUP].nunique(),
                         "evaluable": bool(g[GROUP].nunique() >= 2)})
    return pd.DataFrame(rows).sort_values(["target", "n_runs"], ascending=[True, False])


def main() -> None:
    DST.mkdir(exist_ok=False)
    d = pd.concat([pd.read_csv(SRC / "train_metadata.tsv", sep="\t", dtype=str),
                   pd.read_csv(SRC / "test_metadata.tsv", sep="\t", dtype=str)],
                  ignore_index=True).fillna("")
    hold = set(pd.read_csv(OPT / "optimal_mean_heldout_projects.txt", header=None)[0])
    folds = pd.read_csv(OPT / "optimal_devfolds_assignment.tsv", sep="\t", index_col=0)["fold"]

    te = d[d[GROUP].isin(hold)].reset_index(drop=True)
    tr = d[~d[GROUP].isin(hold)].reset_index(drop=True)

    assert len(tr) == 2775 and len(te) == 863, (len(tr), len(te))
    assert tr[GROUP].nunique() == 67 and te[GROUP].nunique() == 34 + 1, \
        (tr[GROUP].nunique(), te[GROUP].nunique())
    assert not (set(tr[GROUP]) & set(te[GROUP]))
    assert not set(tr.Run_accession) & set(te.Run_accession)

    elig = class_eligibility(d)
    for _, r in elig[elig.evaluable].iterrows():
        a = (tr[r.target] == r["class"]).sum()
        b = (te[r.target] == r["class"]).sum()
        assert a > 0 and b > 0, f"{r.target}/{r['class']} missing on one side ({a}, {b})"

    dev = tr[["Run_accession", GROUP]].copy()
    dev["fold"] = dev[GROUP].map(folds)
    assert dev.fold.notna().all(), "training project without a fold"
    sizes = dev.fold.value_counts(normalize=True)
    assert sizes.between(0.17, 0.23).all(), sizes.to_dict()
    for t in TASKS:
        sub = tr[tr[t] != ""]
        for cls, g in sub.groupby(t):
            if g[GROUP].nunique() >= 2:
                spanned = dev[dev[GROUP].isin(g[GROUP].unique())].fold.nunique()
                assert spanned >= 2, f"{t}/{cls} confined to one fold"

    tr.to_csv(DST / "train_metadata.tsv", sep="\t", index=False)
    te.to_csv(DST / "test_metadata.tsv", sep="\t", index=False)
    tr.Run_accession.to_csv(DST / "train_accessions.txt", index=False, header=False)
    te.Run_accession.to_csv(DST / "test_accessions.txt", index=False, header=False)
    dev[["Run_accession", "fold"]].to_csv(DST / "dev_folds.tsv", sep="\t", index=False)
    class_eligibility(d).to_csv(DST / "class_eligibility.tsv", sep="\t", index=False)

    cfg = {
        "version": "v17",
        "method": "integer-programming optimal whole-BioProject assignment; see PROTOCOLS.md v17",
        "criterion": "minimise mean |held-out class share - 25%| over the eligible classes; "
                     "held-out 23-27% of runs; both sides >= 30 projects; every >=2-project "
                     "class on both sides; proven optimal (HiGHS, zero gap)",
        "labels": "v9 metadata, verified identical to AMD v26.03.0",
        "counts": {"train": len(tr), "test": len(te)},
        "bioprojects": {"train": int(tr[GROUP].nunique()), "test": int(te[GROUP].nunique())},
        "dev_folds": "optimal 5-way grouped partition, same criterion vs 20% per fold",
        "assignments": ["results/split_optim_v17/optimal_mean_heldout_projects.txt",
                        "results/split_optim_v17/optimal_devfolds_assignment.tsv"],
        "built": "2026-10-07",
    }
    (DST / "split_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    log.info("splits_v17 written: train %d/%d projects, held-out %d/%d projects; all assertions passed",
             len(tr), tr[GROUP].nunique(), len(te), te[GROUP].nunique())


if __name__ == "__main__":
    sys.exit(main())
