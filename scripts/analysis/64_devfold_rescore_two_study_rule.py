#!/usr/bin/env python3
"""Dev-fold scores with the eligibility rule applied inside the training side.

The eligibility rule (>= 2 BioProjects) exists because a single-study class cannot
sit on both sides of a study-grouped split and scores 0 for a structural reason.
Inside the dev folds the same holds for a class with only one TRAINING study: when
that study is the evaluation fold, the class has no training example. This script
re-scores with dev-eligible = evaluable AND >= 2 training BioProjects, next to the
current rule, for: (1) pooled dev-fold macro-F1 per model, old and new split;
(2) the paired single-vs-shared-network test; (3) the paired DIANA-vs-network test.

    ./env/bin/python scripts/analysis/64_devfold_rescore_two_study_rule.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/devfold_rescore_two_study_v17"
TASKS = ["community_type", "feature", "sample_host", "material"]
N_BOOT = 2000


def rng_for(task: str, name: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{task}:{name}".encode()).hexdigest()[:16], 16))


def f1_elig(y: np.ndarray, p: np.ndarray, eligible: set) -> float:
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def eligibility(split: str) -> dict[str, dict[str, set]]:
    el = pd.read_csv(ROOT / f"data/splits_{split}/class_eligibility.tsv", sep="\t")
    tr = pd.read_csv(ROOT / f"data/splits_{split}/train_metadata.tsv", sep="\t", low_memory=False)
    out = {}
    for t in TASKS:
        ev = set(el[(el.target == t) & el.evaluable]["class"].astype(str))
        n = tr[tr[t].notna()].groupby(tr[t].astype(str))["archive_project"].nunique()
        out[t] = {"current": ev, "two-study": {c for c in ev if n.get(c, 0) >= 2}}
    return out


def from_probs(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d = d[d.y_true.notna()].set_index("Run_accession")
    pc = [c for c in d.columns if c.startswith("p_")]
    return pd.DataFrame({"y": d.y_true.astype(str),
                         "pred": [pc[i][2:] for i in d[pc].to_numpy().argmax(1)]}, index=d.index)


def from_oof(path: Path, task: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d = d[d[f"{task}_true"].notna()].set_index("Run_accession")
    return pd.DataFrame({"y": d[f"{task}_true"].astype(str), "pred": d[f"{task}_pred"].astype(str)}, index=d.index)


def paired(a: pd.DataFrame, b: pd.DataFrame, project: pd.Series, elig: set, task: str, name: str) -> dict:
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y, pa, pb = a.y.to_numpy(), a.pred.to_numpy(), b.pred.to_numpy()
    g = project.loc[common].astype(str).to_numpy()
    uniq = np.unique(g)
    by = {q: np.where(g == q)[0] for q in uniq}
    rng = rng_for(task, name)
    dd = []
    for _ in range(N_BOOT):
        sel = np.concatenate([by[q] for q in rng.choice(uniq, size=len(uniq), replace=True)])
        dd.append(f1_elig(y[sel], pa[sel], elig) - f1_elig(y[sel], pb[sel], elig))
    dd = np.asarray(dd, float)
    dd = dd[np.isfinite(dd)]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    return {"a": round(f1_elig(y, pa, elig), 3), "b": round(f1_elig(y, pb, elig), 3),
            "delta": round(f1_elig(y, pa, elig) - f1_elig(y, pb, elig), 3),
            "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "verdict": "first" if lo > 0 else ("second" if hi < 0 else "tie")}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    E = {"old": eligibility("v9"), "new": eligibility("v17")}

    rows = []
    for split, v in [("old", "v9"), ("new", "v17")]:
        for t in TASKS:
            models = {m: from_probs(ROOT / f"results/devfold_probs_{v}/probs_{m}_{t}.tsv")
                      for m in ["LogisticRegression_Bal", "RandomForest_Bal", "kNN"]}
            if split == "old":
                models["networks (4 separate)"] = from_oof(ROOT / f"results/reference_v9_oof/oof_{t}_{t}.tsv", t)
            else:
                models["shared network"] = from_probs(ROOT / f"results/devfold_probs_v17/probs_Multitask_{t}.tsv")
                models["DIANA (shared network + logreg)"] = from_probs(ROOT / f"results/devfold_probs_v17/probs_DIANA_{t}.tsv")
            for m, d in models.items():
                for rule in ["current", "two-study"]:
                    el = E[split][t][rule]
                    rows.append({"split": split, "task": t, "model": m, "rule": rule,
                                 "n_classes": len(el & set(d.y)),
                                 "f1": round(f1_elig(d.y.to_numpy(), d.pred.to_numpy(), el), 3)})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "devfold_scores_both_rules.tsv", sep="\t", index=False)

    meta = pd.read_csv(ROOT / "data/splits_v17/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    prows = []
    for t in TASKS:
        single = from_oof(ROOT / f"results/seeds_ensemble_v17/oof_{t}_{t}.tsv", t)
        multi = from_oof(ROOT / f"results/seeds_ensemble_v17/oof_multitask_{t}.tsv", t)
        diana = from_probs(ROOT / f"results/devfold_probs_v17/probs_DIANA_{t}.tsv")
        net = from_probs(ROOT / f"results/devfold_probs_v17/probs_Multitask_{t}.tsv")
        for rule in ["current", "two-study"]:
            el = E["new"][t][rule]
            prows.append({"test": "4 separate minus shared", "task": t, "rule": rule,
                          **paired(single, multi, meta.archive_project, el, t, "a6:single-multi")})
            prows.append({"test": "DIANA minus shared network", "task": t, "rule": rule,
                          **paired(diana, net, meta.archive_project, el, t, "diana-vs-net")})
    pt = pd.DataFrame(prows)
    pt.to_csv(OUT / "paired_tests_both_rules.tsv", sep="\t", index=False)

    print("pooled dev-fold macro-F1, current rule vs two-study rule:")
    print(scores.pivot_table(index=["split", "model"], columns=["task", "rule"], values="f1").round(3).to_string())
    print("\nclasses counted per task (new split): current vs two-study")
    print(scores[(scores.split == "new") & (scores.model == "shared network")][["task", "rule", "n_classes"]].to_string(index=False))
    print("\npaired tests on the new split (first minus second, 2,000 BioProject resamples):")
    print(pt.to_string(index=False))


if __name__ == "__main__":
    main()
