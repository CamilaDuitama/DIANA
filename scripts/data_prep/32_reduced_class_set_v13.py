#!/usr/bin/env python3
"""K1: the v13 class set. A class is modelled only if it has >= 2 training BioProjects.

Rule fixed 2026-09-30 before any fit (PROJECT.md section 8, K protocol). Reason: on
the dev folds no model reaches F1 0.5 on any of the 11 classes backed by a single
training study (results/class_support_v9/), and the v9 eligibility table counted
BioProjects over train and held-out together, so 12 of its 38 eligible classes had one
training study or none.

What this script writes, data/splits_v13/:
  class_eligibility.tsv  per task and class: n_runs and n_bioprojects counted on the
                         TRAINING runs, evaluable = n_bioprojects >= 2, plus the held-out
                         counts and the v9 flag for the record
  train_metadata.tsv     v9 training metadata with the label of every non-evaluable
                         class set to missing, per task; every other column unchanged
  test_metadata.tsv      copied unchanged: held-out runs of a dropped class are
                         out-of-vocabulary at evaluation time and are counted, not scored
  dev_folds.tsv, train_accessions.txt, test_accessions.txt, split_config.json
                         copied unchanged (same split, same folds, same held-out set)
  class_set_v13.json     the rule, the counts, the source files

and the configs:
  configs/final_fixed_v13/<arm>.json    configs/final_fixed_v9/<arm>.json with the v13
                                        metadata and train-ids paths; hyperparameters,
                                        features and seed unchanged
  configs/sig_v13/<arm>_fold<k>.json    the same on results/s8_signatures_v9/sig_fold<k>.npz

Refuses to run if data/splits_v13/ exists.

    ./env/bin/python scripts/data_prep/32_reduced_class_set_v13.py
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SRC, DST = ROOT / "data/splits_v9", ROOT / "data/splits_v13"
CFG_SRC, CFG_DST, CFG_SIG = ROOT / "configs/final_fixed_v9", ROOT / "configs/final_fixed_v13", ROOT / "configs/sig_v13"
SIG_NPZ = "results/s8_signatures_v9/sig_fold{k}.npz"
TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = ["multitask", "community_type", "feature", "sample_host", "material"]
MIN_TRAIN_PROJECTS = 2
COPY = ["test_metadata.tsv", "dev_folds.tsv", "train_accessions.txt", "test_accessions.txt", "split_config.json"]


def eligibility(train: pd.DataFrame, test: pd.DataFrame, v9: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TASKS:
        v9t = v9[v9.target == t].set_index(v9[v9.target == t]["class"].astype(str))
        classes = sorted(set(train[t].dropna().astype(str)) | set(test[t].dropna().astype(str)))
        for c in classes:
            tr, te = train[train[t].astype(str) == c], test[test[t].astype(str) == c]
            n_bp = int(tr.archive_project.nunique())
            rows.append({"target": t, "class": c, "n_runs": len(tr), "n_bioprojects": n_bp,
                         "evaluable": bool(n_bp >= MIN_TRAIN_PROJECTS),
                         "n_heldout_runs": len(te), "n_heldout_bioprojects": int(te.archive_project.nunique()),
                         "evaluable_v9": bool(v9t.evaluable.get(c, False))})
    return pd.DataFrame(rows).sort_values(["target", "n_runs"], ascending=[True, False]).reset_index(drop=True)


def mask_labels(train: pd.DataFrame, elig: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = train.copy()
    counts = {}
    for t in TASKS:
        keep = set(elig[(elig.target == t) & elig.evaluable]["class"].astype(str))
        labelled = train[t].notna()
        drop = labelled & ~train[t].astype(str).isin(keep)
        counts[t] = {"classes_kept": len(keep), "classes_dropped": int(elig[(elig.target == t) & ~elig.evaluable].n_runs.gt(0).sum()),
                     "runs_labelled_before": int(labelled.sum()), "runs_masked": int(drop.sum()),
                     "runs_labelled_after": int((labelled & ~drop).sum())}
        train.loc[drop, t] = np.nan
        left = train[t].dropna().astype(str)
        n_bp = train.loc[left.index].groupby(left).archive_project.nunique()
        if (n_bp < MIN_TRAIN_PROJECTS).any():
            raise AssertionError(f"{t}: a kept class has < {MIN_TRAIN_PROJECTS} training BioProjects")
        if set(left) != keep:
            raise AssertionError(f"{t}: labels left after masking differ from the kept classes")
    return train, counts


def write_configs(counts: dict) -> None:
    for d in (CFG_DST, CFG_SIG):
        if d.exists():
            raise SystemExit(f"{d} exists; refusing to overwrite")
        d.mkdir(parents=True)
    note = {"k_protocol": f"K1 ({date.today().isoformat()}): v13 class set, a class is modelled only if it has "
                          f">= {MIN_TRAIN_PROJECTS} training BioProjects (data/splits_v13/class_eligibility.tsv). "
                          "Hyperparameters, features, folds, held-out set and seed unchanged from configs/final_fixed_v9/.",
            "classes_kept_per_task": {t: counts[t]["classes_kept"] for t in TASKS}}
    for arm in ARMS:
        cfg = json.load(open(CFG_SRC / f"{arm}.json"))
        cfg["metadata_path"] = "data/splits_v13/train_metadata.tsv"
        cfg["train_ids_path"] = "data/splits_v13/train_accessions.txt"
        cfg["output_dir"] = f"results/final_fixed_v13/{arm}/final_model"
        cfg["_provenance"] = {**cfg.get("_provenance", {}), **note}
        json.dump(cfg, open(CFG_DST / f"{arm}.json", "w"), indent=2)
        for k in range(5):
            sig = dict(cfg)
            sig["features_path"] = SIG_NPZ.format(k=k)
            sig["output_dir"] = f"results/epoch_budget_sig_v13/{arm}/fold{k}"
            sig["_provenance"] = {**cfg["_provenance"], "s8_8": "fractions + the 38 per-fold class-signature columns of "
                                  "S8.8 (results/s8_signatures_v9/), unchanged; the columns of dropped classes stay as features."}
            json.dump(sig, open(CFG_SIG / f"{arm}_fold{k}.json", "w"), indent=2)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if DST.exists():
        raise SystemExit(f"{DST} exists; refusing to overwrite (delete it deliberately first)")
    train = pd.read_csv(SRC / "train_metadata.tsv", sep="\t", low_memory=False)
    test = pd.read_csv(SRC / "test_metadata.tsv", sep="\t", low_memory=False)
    v9 = pd.read_csv(SRC / "class_eligibility.tsv", sep="\t")
    elig = eligibility(train, test, v9)
    masked, counts = mask_labels(train, elig)
    DST.mkdir(parents=True)
    elig.to_csv(DST / "class_eligibility.tsv", sep="\t", index=False)
    masked.to_csv(DST / "train_metadata.tsv", sep="\t", index=False)
    for f in COPY:
        shutil.copy2(SRC / f, DST / f)
    back = pd.read_csv(DST / "train_metadata.tsv", sep="\t", low_memory=False)
    if len(back) != len(train) or list(back.Run_accession) != list(train.Run_accession):
        raise AssertionError("masked metadata lost rows or changed order")
    for t in TASKS:
        if back[t].notna().sum() != counts[t]["runs_labelled_after"]:
            raise AssertionError(f"{t}: label count after round trip differs")
    dropped = elig[~elig.evaluable & (elig.n_runs > 0)]
    record = {"rule": f"a class is modelled and scored only if it has >= {MIN_TRAIN_PROJECTS} training BioProjects",
              "fixed_on": date.today().isoformat(), "source": str(SRC.relative_to(ROOT)),
              "unchanged": COPY, "counts": counts,
              "eligible_classes_total": int(elig.evaluable.sum()), "eligible_classes_v9": int(elig.evaluable_v9.sum()),
              "dropped_classes_with_training_runs": dropped[["target", "class", "n_runs", "n_bioprojects"]].to_dict("records"),
              "heldout_runs_of_dropped_classes": {t: int(elig[(elig.target == t) & ~elig.evaluable].n_heldout_runs.sum()) for t in TASKS}}
    json.dump(record, open(DST / "class_set_v13.json", "w"), indent=2)
    write_configs(counts)
    for t in TASKS:
        logger.info("%s: %s", t, counts[t])
    logger.info("eligible classes: %d (v9 table: %d); dropped classes with training runs: %d",
                record["eligible_classes_total"], record["eligible_classes_v9"], len(dropped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
