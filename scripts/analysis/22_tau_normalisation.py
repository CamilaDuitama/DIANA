#!/usr/bin/env python3
"""E1: tau-normalisation of the classifier, validated on the dev folds.

The decoupling result (Kang et al. 2020) is that a representation learned under class
imbalance is already good and the damage is concentrated in the classifier: the
weight norms of frequent classes grow larger, so the argmax is biased toward them.
tau-normalisation divides each class's weight vector by its own norm raised to tau,
which removes that bias without retraining anything.

DIANA's profile matches the signature: 0.95 macro-F1 on train and collapse on rare
classes held-out. And unlike the label-shift correction (E2), this assumes nothing
about how train and test differ, which matters because E2's failure showed the shift
here is joint rather than label-only.

Validated out-of-fold: the classifier of each dev-fold model is renormalised and
re-applied to the fold it never saw. Held-out is not touched. tau would be selected
on the dev folds and only then applied once.

    ./env/bin/python scripts/analysis/22_tau_normalisation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TASKS = ["community_type", "feature", "sample_host", "material"]
TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_fold_model(task: str, fold: int, input_dim: int):
    from diana.models.multitask_mlp import MultiTaskMLP
    base = PROJECT_ROOT / f"results/calibration_v9/fit_{task}_fold{fold}"
    cfg = json.loads((base / "final_training_config.json").read_text())
    mp = cfg["hyperparameters"]["model_params"]
    enc = json.loads((base / "label_encoders.json").read_text())
    cls = enc[task]["classes"] if isinstance(enc[task], dict) else enc[task]
    model = MultiTaskMLP(input_dim=input_dim, hidden_dims=mp["hidden_dims"],
                         num_classes={task: len(cls)}, regression_tasks=[],
                         dropout=mp["dropout"], use_batch_norm=mp["use_batch_norm"],
                         activation=mp["activation"])
    sd = torch.load(base / "best_model.pth", map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd)
    model.eval()
    return model, [str(c) for c in cls]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/tau_norm_v9")
    args = ap.parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
    from load_v9_features import load_features

    md = pd.read_csv(PROJECT_ROOT / "data/splits_v9/train_metadata.tsv",
                     sep="\t", low_memory=False)
    el = pd.read_csv(PROJECT_ROOT / "data/splits_v9/class_eligibility.tsv", sep="\t")
    X, kept = load_features(md.Run_accession)
    md = md.set_index("Run_accession").loc[kept].reset_index()
    pos = {a: i for i, a in enumerate(md.Run_accession.astype(str))}

    rows = []
    for task in TASKS:
        elig = set(el[(el.target == task) & el.evaluable]["class"].astype(str))
        for fold in range(5):
            ids = (PROJECT_ROOT /
                   f"results/calibration_v9/ids/cal_eval_fold{fold}.txt").read_text().split()
            model, cls = load_fold_model(task, fold, X.shape[1])
            head = model.heads[task]
            final = [mod for mod in head if isinstance(mod, torch.nn.Linear)][-1]
            W0 = final.weight.detach().clone()
            b0 = final.bias.detach().clone() if final.bias is not None else None
            norms = W0.norm(dim=1, keepdim=True).clamp_min(1e-12)

            idx = [pos[a] for a in ids if a in pos]
            sub = md.iloc[idx]
            m = sub[task].notna().to_numpy()
            if m.sum() < 20:
                continue
            Xi = X[np.array(idx)[m]]
            y = sub[task].astype(str).to_numpy()[m]
            with torch.no_grad():
                Z = model.backbone(torch.from_numpy(Xi).float())
            keep = sorted(c for c in set(y) if c in elig)
            if not keep:
                continue
            for tau in TAUS:
                with torch.no_grad():
                    final.weight.copy_(W0 / norms.pow(tau))
                    # the bias carries the same frequency bias, so scale it with the row
                    if b0 is not None:
                        final.bias.copy_(b0 / norms.squeeze(1).pow(tau))
                    logits = final(torch.nn.Sequential(*list(head)[:-1])(Z))
                pred = np.array(cls)[logits.argmax(1).numpy()]
                rows.append({"task": task, "fold": fold, "tau": tau,
                             "n": int(m.sum()),
                             "f1": f1_score(y, pred, labels=keep, average="macro",
                                            zero_division=0)})
            with torch.no_grad():
                final.weight.copy_(W0)
                if b0 is not None:
                    final.bias.copy_(b0)
            print(f"  {task:<16}fold {fold} done ({int(m.sum())} scored)")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "tau_sweep.tsv", sep="\t", index=False)
    print("\nMean out-of-fold f1_macro_eligible by tau (tau = 0 is no change):\n")
    piv = df.pivot_table(index="tau", columns="task", values="f1", aggfunc="mean")
    print(piv.round(4).to_string())
    print("\nBest tau per task, and what it buys over tau = 0:\n")
    for task in TASKS:
        g = piv[task] if task in piv.columns else None
        if g is None:
            continue
        base = g.loc[0.0]
        best_t = g.idxmax()
        print(f"  {task:<16}tau={best_t:.1f}  {base:.4f} -> {g.max():.4f}  "
              f"({g.max()-base:+.4f})")
    print(f"\nwrote {args.output / 'tau_sweep.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
