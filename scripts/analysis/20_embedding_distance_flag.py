#!/usr/bin/env python3
"""D1: flag a run by how far it sits from the training runs of its *stated* class.

Why this signal and not another
-------------------------------
The reported detector scores `1 - P(stated label)`. Table 7 shows logistic regression
matches DIANA on that, and the temperature test shows the small remaining gap is not
a calibration artefact, so no amount of rescaling recovers it. This is a different
question: **does this sample look like the other samples carrying this label?**

It is the one signal a linear model has no analogue for. Logistic regression has no
learned representation, only a weight vector per class, so there is no space in which
to measure "unlike the others". DIANA's backbone gives a low-dimensional space
learned for exactly these four tasks.

It is also independent of `P(stated label)`: a run can be confidently classified and
still sit far from its stated class's training cloud, and vice versa. So the two can
be combined, though this script only measures D1 on its own.

Method
------
Embed every training run and every held-out run with the task's backbone. For a
held-out run whose stated label is `c`, the score is the Mahalanobis-style distance
to the training runs labelled `c`, using the centroid and a shrunk covariance shared
across classes, because per-class covariance is unestimable at 11 training runs.
Falls back to Euclidean distance to the centroid when the shared covariance is
singular. Runs whose stated class has no training runs are scored `inf`, i.e. flagged
outright, which is the D2 gap handled correctly here.

    ./env/bin/python scripts/analysis/20_embedding_distance_flag.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
TASKS = ["community_type", "feature", "sample_host", "material"]
TARGET_FPR = [0.05, 0.10]


def embed(task: str, X: np.ndarray) -> np.ndarray:
    from diana.models.multitask_mlp import MultiTaskMLP
    cfg = json.loads((PROJECT_ROOT /
        f"results/search_v9_final_single_{task}/final_training_config.json").read_text())
    mp = cfg["hyperparameters"]["model_params"]
    enc = json.loads((PROJECT_ROOT /
        f"results/search_v9_final_single_{task}/final_model/label_encoders.json").read_text())
    n_cls = len(enc[task]["classes"] if isinstance(enc[task], dict) else enc[task])
    model = MultiTaskMLP(input_dim=X.shape[1], hidden_dims=mp["hidden_dims"],
                         num_classes={task: n_cls}, regression_tasks=[],
                         dropout=mp["dropout"], use_batch_norm=mp["use_batch_norm"],
                         activation=mp["activation"])
    sd = torch.load(PROJECT_ROOT /
        f"results/search_v9_final_single_{task}/final_model/best_model.pth",
        map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
    model.load_state_dict(sd)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model.backbone(torch.from_numpy(X[i:i+256]).float()).numpy())
    return np.vstack(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planted", type=Path,
                    default=PROJECT_ROOT / "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv")
    ap.add_argument("--shrink", type=float, default=0.1,
                    help="ridge added to the shared covariance, as a fraction of its trace")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/embedding_distance_flag")
    args = ap.parse_args()

    from load_v9_features import load_features
    tr_md = pd.read_csv(PROJECT_ROOT / "data/splits_v9/train_metadata.tsv",
                        sep="\t", low_memory=False)
    te_md = pd.read_csv(PROJECT_ROOT / "data/splits_v9/test_metadata.tsv",
                        sep="\t", low_memory=False)
    plant = pd.read_csv(args.planted, sep="\t")

    Xtr, ktr = load_features(tr_md.Run_accession)
    Xte, kte = load_features(te_md.Run_accession)
    tr_md = tr_md.set_index("Run_accession").loc[ktr].reset_index()
    te_md = te_md.set_index("Run_accession").loc[kte].reset_index()

    rows = []
    for task in TASKS:
        Ztr, Zte = embed(task, Xtr), embed(task, Xte)
        lab = tr_md[task].astype(str)
        have = lab.notna() & (lab != "nan")
        # One shared, shrunk covariance: per-class covariance is unestimable when a
        # class has 11 training runs and the embedding is 192-dimensional.
        C = np.cov(Ztr[have.to_numpy()].T)
        C += np.eye(C.shape[0]) * args.shrink * np.trace(C) / C.shape[0]
        try:
            Ci = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Ci = None
        cent = {c: Ztr[(lab == c).to_numpy() & have.to_numpy()].mean(0)
                for c in lab[have].unique()}

        pl = plant[["Run_accession", task, f"{task}_planted"]].dropna()
        m = te_md.merge(pl, on="Run_accession", suffixes=("", "_st"))
        idx = {a: i for i, a in enumerate(te_md.Run_accession.astype(str))}
        stated = m[f"{task}_st"].astype(str) if f"{task}_st" in m.columns else m[task].astype(str)
        d = []
        for acc, c in zip(m.Run_accession.astype(str), stated):
            if c not in cent or acc not in idx:
                d.append(np.inf); continue
            v = Zte[idx[acc]] - cent[c]
            d.append(float(np.sqrt(v @ Ci @ v)) if Ci is not None else float(np.linalg.norm(v)))
        d = np.array(d)
        y = m[f"{task}_planted"].astype(int).to_numpy()
        finite = np.isfinite(d)
        n_inf = int((~finite).sum()); n_inf_planted = int(y[~finite].sum())
        if y[finite].sum() == 0 or y[finite].sum() == finite.sum():
            print(f"  {task}: one class only after filtering, skipping"); continue
        auc = float(roc_auc_score(y[finite], d[finite]))
        apr = float(average_precision_score(y[finite], d[finite]))
        fpr, tpr, thr = roc_curve(y[finite], d[finite])
        rec = {"task": task, "roc_auc": auc, "average_precision": apr,
               "n_scored": int(finite.sum()), "n_planted": int(y[finite].sum()),
               "n_unscorable_by_probability": n_inf,
               "n_unscorable_that_were_planted": n_inf_planted,
               "embedding_dim": int(Ztr.shape[1])}
        for t in TARGET_FPR:
            i = max(int(np.searchsorted(fpr, t, side="right") - 1), 0)
            rec[f"tpr_at_fpr{int(t*100)}"] = float(tpr[i])
        rows.append(rec)
        print(f"  {task:<16}dim={Ztr.shape[1]:<4}AUC={auc:.3f} AP={apr:.3f} "
              f"TPR@5%={rec['tpr_at_fpr5']:.3f} TPR@10%={rec['tpr_at_fpr10']:.3f} "
              f"(+{n_inf} runs with an unknown stated class, {n_inf_planted} of them planted)")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "embedding_distance.tsv", sep="\t", index=False)
    print(f"\nwrote {args.output / 'embedding_distance.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
