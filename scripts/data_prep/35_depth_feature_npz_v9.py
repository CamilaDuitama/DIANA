#!/usr/bin/env python3
"""X9b: the v9 unitig fractions plus one extra input column, the run's sequencing depth.

The column is log10(1 + number of non-zero unitigs) / log10(1 + 110,202), so it lies in
[0, 1] like the fractions. Nothing else changes. Written in the MatrixLoader npz layout
(frac, feature_ids, sample_ids) to results/x9b_v9/frac_logdepth.npz, and the configs
configs/x9b_v9/<arm>.json point at it (otherwise identical to configs/final_fixed_v9/).

    sbatch on edid, 16 GB: ./env/bin/python scripts/data_prep/35_depth_feature_npz_v9.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz"
OUT = ROOT / "results/x9b_v9/frac_logdepth.npz"
CFG_SRC, CFG_DST = ROOT / "configs/final_fixed_v9", ROOT / "configs/x9b_v9"
ARMS = ["community_type", "feature", "sample_host", "material"]


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    with np.load(SRC, allow_pickle=False) as z:
        ids = z["sample_ids"].astype(str); F = z["frac"].astype(np.float32)
        fid = z["feature_ids"].astype(str)[:F.shape[1]]   # the s8a npz lists the unitig ids first, then the off-list hash ids
    n_nonzero = (F > 0).sum(axis=1)
    depth = (np.log10(1.0 + n_nonzero) / np.log10(1.0 + F.shape[1])).astype(np.float32)
    X = np.concatenate([F, depth[:, None]], axis=1)
    np.savez(OUT, frac=X, feature_ids=np.concatenate([fid, np.array(["log_nonzero_scaled"])]), sample_ids=ids)
    CFG_DST.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        cfg = json.load(open(CFG_SRC / f"{arm}.json"))
        cfg["features_path"] = str(OUT.relative_to(ROOT))
        cfg["output_dir"] = f"results/x9b_v9/{arm}"
        cfg["_provenance"] = {**cfg.get("_provenance", {}),
                              "x9b": f"X9b ({date.today().isoformat()}): fractions + log10(1 + non-zero unitigs) scaled to [0, 1] as one extra "
                                     "input column; trained with the frozen reference recipe (lr, schedule, budget, 5 seeds)."}
        json.dump(cfg, open(CFG_DST / f"{arm}.json", "w"), indent=2)
    print(f"wrote {OUT}: {X.shape}; depth column min {depth.min():.3f} median {np.median(depth):.3f} max {depth.max():.3f}; configs in {CFG_DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
