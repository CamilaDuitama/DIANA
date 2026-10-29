#!/usr/bin/env python3
"""Configs and arm table of the v19 dev-fold screens (PROTOCOLS.md, v19 amendment of 2026-10-21).

Every arm is the final DIANA network (configs/v19_final/multitask.json: settings, 173 epochs, seeds
42, 1, 2, 3, 4, the five dev folds) with one change: its input (a copy of the config pointing at a
results/screens_v19/inputs/ file, one per fold when the input is fitted per fold) or one training
option of 04_epoch_budget_dev_folds.py. Writes configs/screens_v19/*.json and configs/screens_v19/arms.tsv
(arm, config template with {fold}, epoch cap, extra arguments), read by run_screens_v19.sbatch.

    ./env/bin/python scripts/training/screen_configs_v19.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/v19_final/multitask.json"
OUT = ROOT / "configs/screens_v19"
INPUTS = "results/screens_v19/inputs"
LRDIR = "results/screens_v19/lr_inputs"
CAP = 400                                   # the search cap (PROTOCOLS v19 C3), for the early-stopping arms
# arm -> input file (None = the fractions); "{fold}" marks a per-fold input
INPUT_ARMS = {
    "pa": "pa.npz", "abund": "abund_log1p.npz",
    "frac_pca192": "frac_pca192_fold{fold}.npz", "pa_pca192": "pa_pca192_fold{fold}.npz",
    "frac_pca191nd": "frac_pca191nd_fold{fold}.npz", "pa_pca191nd": "pa_pca191nd_fold{fold}.npz",
    "frac_kmer4": "frac_kmer4_fold{fold}.npz", "frac_clust": "frac_clust_fold{fold}.npz",
    "logdepth": "frac_logdepth.npz",
}
OPTION_ARMS = {
    "stop_random": (CAP, "--early-stop random --val-split 0.1 --patience 20 --save-probs-by-epoch"),
    "stop_grouped": (CAP, "--early-stop grouped --val-split 0.1 --patience 20 --save-probs-by-epoch"),
    "lr_init": (None, f"--linear-init-dir {LRDIR}/linear_init"),
    "lr_offsets": (None, f"--offsets-dir {LRDIR}/offsets"),
    "ema": (None, "--ema 0.9"),
}
# Held after the smoke run, then adapted to the shared network (PROTOCOLS.md amendments 2026-10-21, Camila: "2");
# their own table, arms_adapted.tsv, so the 14-arm run already queued is untouched.
ADAPTED = {
    "depth_aug": (None, "--depth-augment 0.5"),
    "mixup": (None, "--mixup 0.4"),
    "adv_conf0.1": (None, "--adversarial 0.1 --adversarial-mode confusion"),
    "adv_conf0.3": (None, "--adversarial 0.3 --adversarial-mode confusion"),
    "adv_conf1.0": (None, "--adversarial 1.0 --adversarial-mode confusion"),
}
# ADV2 (PROTOCOLS.md v19 amendment 2026-10-27): two adversary variants, scored against the network alone; arms_adv.tsv.
ADV2 = {
    "adv_cond0.3": (None, "--adversarial 0.3 --adversarial-mode confusion --adversarial-conditional"),
    "adv_ct0.3": (None, "--adversarial 0.3 --adversarial-mode confusion --adversarial-layer community_type"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = json.load(open(BASE))
    epochs = int(base["epochs"])
    rows = []
    for arm, f in INPUT_ARMS.items():
        folds = range(5) if "{fold}" in f else [None]
        for k in folds:
            cfg = copy.deepcopy(base)
            cfg["features_path"] = f"{INPUTS}/{f.format(fold=k)}" if k is not None else f"{INPUTS}/{f}"
            cfg["_provenance"] = {**cfg["_provenance"], "screen": f"v19 screen {arm} (PROTOCOLS.md 2026-10-21): "
                                  "configs/v19_final/multitask.json with this input, nothing else changed"}
            name = f"{arm}_fold{k}.json" if k is not None else f"{arm}.json"
            json.dump(cfg, open(OUT / name, "w"), indent=1)
        tmpl = f"configs/screens_v19/{arm}_fold{{fold}}.json" if "{fold}" in f else f"configs/screens_v19/{arm}.json"
        rows.append((arm, tmpl, epochs, ""))
    for arm, (cap, extra) in OPTION_ARMS.items():
        rows.append((arm, str(BASE.relative_to(ROOT)), cap or epochs, extra))
    adapted = [(arm, str(BASE.relative_to(ROOT)), cap or epochs, extra) for arm, (cap, extra) in ADAPTED.items()]
    adv2 = [(arm, str(BASE.relative_to(ROOT)), cap or epochs, extra) for arm, (cap, extra) in ADV2.items()]
    for name, table in (("arms.tsv", rows), ("arms_adapted.tsv", adapted), ("arms_adv.tsv", adv2)):
        with open(OUT / name, "w") as fh:
            for r in table:
                fh.write("\t".join(map(str, r)) + "\n")
        print(f"{len(table)} arms -> {OUT / name} ({len(table) * 25} fits)")


if __name__ == "__main__":
    main()
