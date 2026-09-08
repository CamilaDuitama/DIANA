#!/usr/bin/env python
"""Is host-associated vs environmental worth a dedicated model head?

Question answered
-----------------
PROJECT.md proposes a binary `branch` head whose ground truth is free (it is
determined by which AncientMetagenomeDir table a run came from). Its value is as
a QC flag: if AMD says host-associated and the model says environmental, that is
a signal. But a flag is only useful if it is accurate -- the branch implied by
v7's community_type predictions was only 89.8 % accurate, which would mean
flagging ~10 % of correctly-labelled samples.

This measures the ceiling directly with a linear model on the v9
BioProject-disjoint split. If a plain logistic regression is near-perfect, a
dedicated head is cheap and the flag is trustworthy. If it sits near 90 %, the
flag is too noisy to lead with.

Inputs : data/splits_v9/{train,test}_metadata.tsv, v9 features
Outputs: printed metrics; results/branch_separability.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/analysis"))
from load_v9_features import load_features  # noqa: E402


def branch_label(df: pd.DataFrame) -> np.ndarray:
    """host-associated runs carry community_type; environmental ones carry feature."""
    return np.where(df.community_type.notna(), "host",
                    np.where(df.feature.notna(), "env", None))


def main() -> int:
    tr = pd.read_csv(PROJECT_ROOT / "data/splits_v9/train_metadata.tsv", sep="\t")
    te = pd.read_csv(PROJECT_ROOT / "data/splits_v9/test_metadata.tsv", sep="\t")
    tr["b"], te["b"] = branch_label(tr), branch_label(te)
    tr, te = tr.dropna(subset=["b"]), te.dropna(subset=["b"])

    Xtr, ktr = load_features(tr.Run_accession)
    Xte, kte = load_features(te.Run_accession)
    ytr = tr.set_index("Run_accession").loc[ktr, "b"].to_numpy()
    yte = te.set_index("Run_accession").loc[kte, "b"].to_numpy()
    print(f"train {Xtr.shape}  test {Xte.shape}")
    print("train balance:", dict(pd.Series(ytr).value_counts()))
    print("test  balance:", dict(pd.Series(yte).value_counts()))

    model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = float(accuracy_score(yte, pred))
    mf1 = float(f1_score(yte, pred, average="macro"))
    maj = float(pd.Series(yte).value_counts().max() / len(yte))
    cm = confusion_matrix(yte, pred, labels=["env", "host"])
    print("\nhost vs environmental — logistic regression, v9 BioProject-disjoint test")
    print("  accuracy       : %.4f" % acc)
    print("  macro F1       : %.4f" % mf1)
    print("  majority class : %.4f" % maj)
    print("  confusion (rows=true [env,host]):")
    print(cm)
    print("\n  -> false-flag rate if used as a QC flag: %.2f%%" % (100 * (1 - acc)))

    out = PROJECT_ROOT / "results/branch_separability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"accuracy": acc, "macro_f1": mf1, "majority_rate": maj,
               "confusion_env_host": cm.tolist(),
               "n_train": int(len(ytr)), "n_test": int(len(yte))},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
