#!/usr/bin/env python
"""Baselines on the v9 BioProject-disjoint partition (PROJECT.md G7).

Question answered
-----------------
What do the numbers on the new partition actually mean? Until simple models have
been run on v9, there is no way to say whether a retrained network is better --
and v7's numbers are not comparable, because both the partition and the label
space changed.

Metric definitions, which differ from the v7 baseline script in one important way
-----------------------------------------------------------------------------
`f1_macro_eligible` averages only over classes that occur in **>= 2
BioProjects** (PROJECT.md G5). A class confined to one BioProject cannot appear
on both sides of a BioProject-disjoint split, so it scores 0 for a structural
reason rather than a modelling one, and including it makes macro-F1 a measure of
the split rather than of the model. 22 of 60 classes are excluded this way; they
remain in training. `f1_macro_seen` (over classes present in y_true) is reported
alongside so the two definitions can be compared.

Runs whose true class never appears in training are counted as out-of-vocabulary
and excluded from the metrics rather than silently scored wrong (R3.5).

Inputs : data/splits_v9/, v9 features via scripts/analysis/load_v9_features.py
Outputs: results/baseline_comparison_v9/{metrics.json,summary.csv}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/analysis"))
from load_v9_features import load_features  # noqa: E402

TARGETS = ["community_type", "feature", "sample_host", "material"]
SPLITS = PROJECT_ROOT / "data/splits_v9"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_models(seed: int) -> dict:
    return {
        "MajorityClass": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_Bal": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "LinearSVM_Bal": LinearSVC(max_iter=5000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                               n_jobs=-1, random_state=seed),
        "RandomForest_Bal": RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                   class_weight="balanced_subsample",
                                                   n_jobs=-1, random_state=seed),
        "kNN_5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def metrics(y_true, y_pred, eligible: set) -> dict:
    seen = sorted(set(y_true))
    elig = sorted(c for c in seen if c in eligible)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro_seen": float(f1_score(y_true, y_pred, labels=seen,
                                        average="macro", zero_division=0)),
        "f1_macro_eligible": float(f1_score(y_true, y_pred, labels=elig,
                                            average="macro", zero_division=0)) if elig else float("nan"),
        "n_classes_seen": len(seen),
        "n_classes_eligible": len(elig),
    }


def bootstrap_ci(y_true, y_pred, eligible: set, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    elig = sorted(c for c in set(y_true) if c in eligible)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, size=n)
        if elig:
            vals.append(f1_score(y_true[i], y_pred[i], labels=elig,
                                 average="macro", zero_division=0))
    if not vals:
        return {}
    return {"f1_macro_eligible_ci_low": float(np.percentile(vals, 2.5)),
            "f1_macro_eligible_ci_high": float(np.percentile(vals, 97.5))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/baseline_comparison_v9")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t")
    te = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t")
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")

    logger.info("loading features")
    Xtr_all, ktr = load_features(tr.Run_accession)
    Xte_all, kte = load_features(te.Run_accession)
    tr = tr.set_index("Run_accession").loc[ktr].reset_index()
    te = te.set_index("Run_accession").loc[kte].reset_index()
    logger.info("train %s  test %s", Xtr_all.shape, Xte_all.shape)

    models = build_models(args.seed)
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    results, rows = {}, []
    for target in args.targets:
        eligible = set(elig_tbl[(elig_tbl.target == target) & elig_tbl.evaluable]["class"])
        mtr = tr[target].notna().to_numpy()
        mte = te[target].notna().to_numpy()
        ytr_all = tr.loc[mtr, target].astype(str).to_numpy()
        seen_in_train = set(ytr_all)

        # R3.5: runs whose class was never trained on are OOV, not wrong answers.
        yte_all = te.loc[mte, target].astype(str).to_numpy()
        in_vocab = np.array([v in seen_in_train for v in yte_all])
        n_oov = int((~in_vocab).sum())

        Xtr, ytr = Xtr_all[mtr], ytr_all
        Xte, yte = Xte_all[mte][in_vocab], yte_all[in_vocab]
        logger.info("%s: train %d, test %d (+%d out-of-vocabulary excluded), "
                    "%d eligible classes", target, len(ytr), len(yte), n_oov, len(eligible))
        results.setdefault(target, {"n_train": len(ytr), "n_test": len(yte),
                                    "n_out_of_vocabulary": n_oov,
                                    "n_eligible_classes": len(eligible)})

        for name, model in models.items():
            t0 = time.time()
            try:
                model.fit(Xtr, ytr)
                pred_te = model.predict(Xte)
                # Report TRAIN as well. Referee 1 objected that the v5 baselines were
                # compared on the training set only; reporting both shows the
                # train/test gap, which is itself a measure of overfitting and of how
                # much the BioProject-disjoint split costs.
                pred_tr = model.predict(Xtr)
            except Exception as exc:  # a model failing must not lose the rest
                logger.warning("%s / %s failed: %s", target, name, exc)
                continue

            m = metrics(yte, pred_te, eligible)
            m.update(bootstrap_ci(yte, pred_te, eligible, args.n_boot, args.seed))
            m["fit_predict_s"] = round(time.time() - t0, 1)
            m_tr = metrics(ytr, pred_tr, eligible)

            results[target][name] = {"test": m, "train": m_tr}
            rows.append({"model": name, "split": "test", "task": target, **m})
            rows.append({"model": name, "split": "train", "task": target, **m_tr})
            logger.info("  %-24s test: acc=%.3f bal=%.3f f1=%.3f | train: acc=%.3f bal=%.3f f1=%.3f  (%.0fs)",
                        name, m["accuracy"], m["balanced_accuracy"], m["f1_macro_eligible"],
                        m_tr["accuracy"], m_tr["balanced_accuracy"], m_tr["f1_macro_eligible"],
                        m["fit_predict_s"])

    json.dump(results, open(args.output / "metrics.json", "w"), indent=2)
    pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
