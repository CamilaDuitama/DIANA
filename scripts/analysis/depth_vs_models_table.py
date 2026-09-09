#!/usr/bin/env python3
"""R3.7 / R3.8 — is the signal sequencing depth?

Referee 3 argued the classes may be separable only because of sequencing depth:
`--out-frac` is a k-mer *completeness* fraction, which rises monotonically with
depth, and k=31 with min-abundance 2 penalises shallow libraries.

There is no single agreed definition of "depth", so this reports four, side by
side, and never picks a favourite:

  metadata depth   log10(read_count), log10(download_size)   -- SRA bookkeeping
  abundance+cov    log10(total unitig abundance), mean frac  -- both data-derived
  abundance alone  log10(total unitig abundance)             -- genuine depth
  coverage alone   mean k-mer fraction                       -- vocabulary breadth

`coverage alone` is deliberately the harshest: mean fraction is close to a
one-number summary of the features themselves, so it grants depth more power than
a referee would ask for. `sample_host` shows why all four are kept -- neither
abundance nor coverage alone beats the majority baseline there, only the pair does.

The comparison that answers the referee is depth vs the full 110,202-unitig
models, on train and on held-out. Held-out is what matters; train is shown because
a depth model that fits train and collapses on held-out is displaying the
BioProject shift, not a depth signal.

    ./env/bin/python scripts/analysis/depth_vs_models_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPTH = PROJECT_ROOT / "results/depth_only_control/metrics.json"
BASE = PROJECT_ROOT / "results/baseline_comparison_v9/summary.csv"
OUT = PROJECT_ROOT / "results/depth_vs_models"
TASKS = ["community_type", "feature", "sample_host", "material"]
METRIC = "f1_macro_eligible"


def _best_depth(block: dict, split: str) -> float:
    """Best depth-only model for one task, so depth is shown at its strongest."""
    vals = []
    for name, r in block.items():
        if not name.startswith("Depth"):
            continue
        v = r.get(split, r) if isinstance(r, dict) else r
        if isinstance(v, dict) and v.get(METRIC) is not None:
            vals.append(v[METRIC])
    return max(vals) if vals else float("nan")


def _majority(block: dict, split: str) -> float:
    r = block.get("MajorityClass", {})
    v = r.get(split, r) if isinstance(r, dict) else r
    return v.get(METRIC, float("nan")) if isinstance(v, dict) else float("nan")


def main() -> int:
    depth = json.load(open(DEPTH))
    base = pd.read_csv(BASE)

    rows = []
    for task in TASKS:
        for split in ("train", "test"):
            row = {"task": task, "split": "held-out" if split == "test" else "train"}
            first = next(iter(depth.values()))
            row["majority"] = _majority(first.get(task, {}), split)
            for set_name, per_task in depth.items():
                short = set_name.split("[")[0].strip()
                row[short] = _best_depth(per_task.get(task, {}), split)
            b = base[(base.task == task) & (base.split == split)]
            b = b[b.model != "MajorityClass"]
            row["best predictor"] = b[METRIC].max() if len(b) else float("nan")
            row["best predictor (model)"] = (
                b.loc[b[METRIC].idxmax(), "model"] if len(b) else "")

            # DIANA slots in here once the search and final fit are done.
            row["DIANA"] = float("nan")
            rows.append(row)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "depth_vs_models.tsv", sep="\t", index=False, float_format="%.3f")

    depth_cols = [c for c in df.columns
                  if c not in ("task", "split", "majority", "best predictor",
                               "best predictor (model)", "DIANA")]
    lines = [f"R3.7 / R3.8 — depth vs predictors vs DIANA   ({METRIC})", ""]
    head = f"{'task':<16}{'split':<10}{'major':>8}"
    head += "".join(f"{c[:15]:>17}" for c in depth_cols)
    head += f"{'PREDICTOR':>11}{'DIANA':>8}"
    lines += [head, "-" * len(head)]
    for _, r in df.iterrows():
        line = f"{r.task:<16}{r.split:<10}{r.majority:>8.3f}"
        line += "".join(f"{r[c]:>17.3f}" for c in depth_cols)
        line += f"{r['best predictor']:>11.3f}"
        line += f"{r['DIANA']:>8.3f}" if pd.notna(r["DIANA"]) else f"{'pending':>8}"
        lines.append(line)

    held = df[df.split == "held-out"]
    ratios = (held["best predictor"] / held[depth_cols].max(axis=1)).dropna()
    lines += ["",
              f"On held-out, the full features beat the strongest depth definition by "
              f"{ratios.min():.1f}-{ratios.max():.1f}x.",
              "Depth does beat the majority baseline, so it is not signal-free -- the",
              "claim is that it is not what the model is using."]

    report = "\n".join(lines)
    print(report)
    (OUT / "summary.txt").write_text(report + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
