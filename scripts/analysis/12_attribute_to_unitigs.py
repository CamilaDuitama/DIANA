#!/usr/bin/env python3
"""Which unitigs drive each class prediction? (R3.11, R1.1)

Referees ask what biological signal the model actually uses. The inputs are unitigs
-- real sequences -- so an attribution over inputs is directly interpretable: rank
the unitigs by how much they push a class logit, pull their sequences out of
`unitigs.fa`, and BLAST them.

Two attribution methods, because they fail differently:

  gradient x input   cheap, one backward pass, but reads the local slope only and
                     saturates -- a feature already driving the logit hard can show
                     near-zero gradient.
  integrated grads   averages the gradient along a straight path from a baseline to
                     the sample, so it does not saturate. The baseline here is the
                     all-zero vector, which for k-mer fractions genuinely means
                     "no sequence observed", so it is a meaningful reference rather
                     than an arbitrary one.

Attributions are averaged over the samples of a class and only over samples the
model gets RIGHT, so the ranking describes what the model uses when it works rather
than what it does when confused.

Feature index -> unitig id is safe because `unitigs.fa` header order is
byte-identical to the matrix row order (verified; see PROJECT.md §3).

    ./env/bin/python scripts/analysis/12_attribute_to_unitigs.py \\
        --model results/search_v9_final/final_model/best_model.pth \\
        --config results/search_v9_final/final_training_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
MATRIX_DIR = PROJECT_ROOT / "data/matrices/matrix_v9_train"


def unitig_ids_and_seqs(fasta: Path) -> tuple[list[str], list[str]]:
    ids, seqs, cur = [], [], []
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
                ids.append(line[1:].split()[0])
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return ids, seqs


def integrated_gradients(model, x, task, cls, steps=32):
    """Mean gradient along the straight path from the all-zero baseline to x."""
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (x * a).clone().requires_grad_(True)
        logit = model(xi)[task][:, cls].sum()
        g, = torch.autograd.grad(logit, xi)
        total += g
    return (x * total / steps).detach()


def grad_x_input(model, x, task, cls):
    xi = x.clone().requires_grad_(True)
    logit = model(xi)[task][:, cls].sum()
    g, = torch.autograd.grad(logit, xi)
    return (xi * g).detach()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--method", choices=["ig", "gxi", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-samples-per-class", type=int, default=40)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/attribution_v9")
    args = ap.parse_args()

    from diana.data.loader import MatrixLoader
    from diana.models.multitask_mlp import MultiTaskMLP

    cfg = json.loads(args.config.read_text())
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    hp = cfg["hyperparameters"]

    feats, ids, _ = MatrixLoader(Path(cfg["features_path"])).load()
    md = (pd.read_csv(cfg["metadata_path"], sep="\t", low_memory=False)
          .set_index("Run_accession").loc[list(ids)].reset_index())

    n_cls = {t: sd[f"heads.{t}.3.weight"].shape[0] for t in cfg["task_names"]}
    model = MultiTaskMLP(input_dim=feats.shape[1],
                         hidden_dims=hp["model_params"]["hidden_dims"],
                         num_classes=n_cls, regression_tasks=[],
                         dropout=hp["model_params"]["dropout"],
                         use_batch_norm=hp["model_params"]["use_batch_norm"],
                         activation=hp["model_params"]["activation"])
    model.load_state_dict(sd)
    model.eval().to(args.device)

    enc_path = args.model.parent / "label_encoders.json"
    encoders = json.loads(enc_path.read_text())

    u_ids, u_seqs = unitig_ids_and_seqs(MATRIX_DIR / "unitigs.fa")
    if len(u_ids) != feats.shape[1]:
        raise AssertionError(
            f"{len(u_ids)} unitigs in unitigs.fa but {feats.shape[1]} features; "
            "index -> unitig mapping would be wrong")

    args.output.mkdir(parents=True, exist_ok=True)
    methods = ["ig", "gxi"] if args.method == "both" else [args.method]
    rows, fasta_lines = [], []

    for task in cfg["task_names"]:
        classes = np.array(encoders[task]["classes"])
        y = md[task].to_numpy()
        for ci, cname in enumerate(classes):
            idx = np.where(y == cname)[0]
            if len(idx) == 0:
                continue
            X = torch.tensor(feats[idx], dtype=torch.float32, device=args.device)
            with torch.no_grad():
                correct = (model(X)[task].argmax(1).cpu().numpy() == ci)
            keep = idx[correct][: args.max_samples_per_class]
            if len(keep) == 0:
                print(f"  {task}/{cname}: 0 correctly predicted, skipped")
                continue
            Xc = torch.tensor(feats[keep], dtype=torch.float32, device=args.device)
            for m in methods:
                att = (integrated_gradients(model, Xc, task, ci, args.ig_steps)
                       if m == "ig" else grad_x_input(model, Xc, task, ci))
                score = att.mean(0).cpu().numpy()
                order = np.argsort(-np.abs(score))[: args.top_k]
                for rank, j in enumerate(order, 1):
                    rows.append({"task": task, "class": cname, "method": m,
                                 "rank": rank, "feature_index": int(j),
                                 "unitig_id": u_ids[j],
                                 "attribution": float(score[j]),
                                 "unitig_len": len(u_seqs[j]),
                                 "n_samples_used": len(keep)})
                    if m == methods[0]:
                        fasta_lines.append(
                            f">{task}|{cname}|rank{rank}|unitig{u_ids[j]}\n{u_seqs[j]}")
            print(f"  {task}/{cname}: {len(keep)} samples, top-{args.top_k} attributed")

    df = pd.DataFrame(rows)
    df.to_csv(args.output / "attributions.tsv", sep="\t", index=False)
    (args.output / "top_unitigs_for_blast.fasta").write_text("\n".join(fasta_lines) + "\n")

    print(f"\nwrote {args.output}")
    print(f"  attributions.tsv           {len(df)} rows")
    print(f"  top_unitigs_for_blast.fasta {len(fasta_lines)} sequences")
    if {"ig", "gxi"} <= set(df.method):
        ov = []
        for (t, c), g in df.groupby(["task", "class"]):
            a = set(g[g.method == "ig"].feature_index)
            b = set(g[g.method == "gxi"].feature_index)
            if a and b:
                ov.append(len(a & b) / len(a | b))
        print(f"  ig vs gradient-x-input top-{args.top_k} Jaccard: "
              f"mean {np.mean(ov):.3f} — low agreement means saturation matters, "
              f"so quote integrated gradients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
