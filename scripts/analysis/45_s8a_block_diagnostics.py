#!/usr/bin/env python3
"""What is in the off-list block: study fingerprints, class signal, or neither?

For each of the K kept off-list hashes (>= 14 training samples, >= 2 projects), on the
2,716 training sketches only:
  * study concentration: the share of the hash's occurrences that fall in its single
    most frequent BioProject (1.0 = one study, 1/n = spread evenly over n studies);
  * per task, class purity: the share of the hash's labelled occurrences that fall in
    its most frequent class, against that class's overall prevalence; a hash is
    "class-enriched" when purity >= 0.9 for a class that is not the task's majority;
  * for class-enriched hashes, whether the class's occurrences come from >= 2 of that
    class's own training projects (the only way it could transfer to a new study).
Also: positives per column and columns per sample, which is what a model has to learn
a weight from. No labels from held-out, no model; a description of the input.

    ./env/bin/python scripts/analysis/45_s8a_block_diagnostics.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "data/sketches_v9"
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/s8a_probe_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]


def main() -> int:
    t0 = time.time()
    with np.load(ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz", allow_pickle=False) as z:
        B = z["block"]  # (2716, K) uint8
        ids = list(z["sample_ids"].astype(str))
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
    proj = pd.factorize(meta["archive_project"].astype(str))[0]
    n_proj = proj.max() + 1
    K = B.shape[1]
    n_pos = B.sum(0).astype(np.int64)
    print(f"block {B.shape}, positives per column: median {np.median(n_pos):.0f}, "
          f"p90 {np.percentile(n_pos, 90):.0f}; columns per sample: median {np.median(B.sum(1)):.0f}, "
          f"p10 {np.percentile(B.sum(1), 10):.0f}, p90 {np.percentile(B.sum(1), 90):.0f}  ({time.time()-t0:.0f} s)")

    # occurrences per (column, project): (n_proj, K) via one-hot projects
    P = np.zeros((n_proj, len(ids)), dtype=np.float32)
    P[proj, np.arange(len(ids))] = 1
    cnt = P @ B.astype(np.float32)          # (n_proj, K)
    top_share = cnt.max(0) / np.maximum(n_pos, 1)
    n_proj_per = (cnt > 0).sum(0)
    summary = {"K": int(K), "median_positives_per_column": float(np.median(n_pos)),
               "median_columns_per_sample": float(np.median(B.sum(1))),
               "top_project_share": {q: float(np.percentile(top_share, q)) for q in (10, 25, 50, 75, 90)},
               "share_of_columns_with_top_project_share_>=0.8": float((top_share >= 0.8).mean()),
               "share_of_columns_with_top_project_share_>=0.5": float((top_share >= 0.5).mean()),
               "median_projects_per_column": float(np.median(n_proj_per))}
    print(json.dumps(summary, indent=2))

    rows = []
    for task in TASKS:
        lab = meta[task].astype(object).to_numpy()
        ok = pd.notna(lab)
        classes, cidx = np.unique(lab[ok].astype(str), return_inverse=True)
        C = np.zeros((len(classes), int(ok.sum())), dtype=np.float32)
        C[cidx, np.arange(int(ok.sum()))] = 1
        cc = C @ B[ok].astype(np.float32)   # (n_classes, K): labelled occurrences per class
        tot = cc.sum(0)
        purity = cc.max(0) / np.maximum(tot, 1)
        best = cc.argmax(0)
        prevalence = C.sum(1) / C.shape[1]
        majority = int(np.argmax(prevalence))
        enriched = (purity >= 0.9) & (tot >= 10) & (best != majority)
        # for enriched hashes: projects of that class in which the hash occurs
        cls_proj = {}
        for c in range(len(classes)):
            m = np.zeros(len(ids), bool); m[np.where(ok)[0][cidx == c]] = True
            Pc = P[:, m] @ B[m].astype(np.float32)   # (n_proj, K)
            cls_proj[c] = (Pc > 0).sum(0)
        n_cls_proj = np.array([cls_proj[best[k]][k] if enriched[k] else 0 for k in range(K)])
        n_cls_proj_total = {c: int(len(set(proj[np.where(ok)[0][cidx == c]]))) for c in range(len(classes))}
        for c in range(len(classes)):
            sel = enriched & (best == c)
            rows.append({"task": task, "class": classes[c], "train_runs": int(C[c].sum()),
                         "train_projects": n_cls_proj_total[c], "enriched_hashes": int(sel.sum()),
                         "enriched_in_>=2_class_projects": int((sel & (n_cls_proj >= 2)).sum()),
                         "median_top_project_share": float(np.median(top_share[sel])) if sel.any() else np.nan})
        print(f"{task}: {int(enriched.sum())} class-enriched hashes (purity >= 0.9, >= 10 labelled occurrences, "
              f"not the majority class); {int((enriched & (n_cls_proj >= 2)).sum())} of them span >= 2 of the class's projects")
    df = pd.DataFrame(rows).sort_values(["task", "train_runs"])
    df.to_csv(OUT / "block_diagnostics_by_class.tsv", sep="\t", index=False)
    json.dump(summary, open(OUT / "block_diagnostics.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"done in {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
