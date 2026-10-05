#!/usr/bin/env python3
"""Which model average is best? All pairwise and three-way probability averages of the final
DIANA, logistic regression, random forest and k-NN, on the grouped dev folds (out-of-fold),
scored on classification (f1_macro_eligible) and flagging (choosing score over the 20
single-label plantings), each combination paired against the pre-declared DIANA + logistic
regression average over BioProjects. Dev-fold scaffolding: choosing a different combination
would need a fresh held-out set; read 11 only confirms the pre-declared one.

    ./env/bin/python scripts/analysis/64_model_average_combos.py
"""
from __future__ import annotations

import glob
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
F = __import__("59_devfold_flagging_score")
from diana.evaluation.metrics import load_eligible_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
OUT = ROOT / "results/model_average_combos_v9"
N_BOOT = 1000


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    plants = [pd.read_csv(f, sep="\t").set_index("Run_accession")
              for f in sorted(glob.glob(str(ROOT / "results/planted_mislabels_dev_v9/planted_train_mixed_r0.1_s*.tsv")))]
    rows, pairs = [], []
    for task in TASKS:
        base = {"DIANA": F.reference_probs(task, ROOT / ("results/step4_2_studyweight_v9" if task == "feature" else "results/reference_v9")),
                "LogReg": F.baseline_probs("LogisticRegression_Bal", task), "RF": F.baseline_probs("RandomForest_Bal", task),
                "kNN": F.baseline_probs("kNN", task)}
        combos = {}
        for r in (2, 3):
            for names in combinations(base, r):
                idx = base[names[0]].index
                cols = set(base[names[0]].columns)
                for n in names[1:]:
                    idx = idx.intersection(base[n].index); cols |= set(base[n].columns)
                cols = sorted(cols)
                combos["+".join(names)] = sum(P.reindex(index=idx, columns=cols).fillna(0.0) for n, P in ((n, base[n]) for n in names)) / r
        models = {**base, **combos}
        eligible = load_eligible_classes(ROOT / "data/splits_v9/class_eligibility.tsv", task)
        projects = np.array(sorted(meta.loc[meta[task].notna(), "archive_project"].astype(str).unique()))
        draws = [rng_for("combo", task, i).integers(0, len(projects), size=len(projects)) for i in range(N_BOOT)]
        boot = {}
        for name, P in models.items():
            lab = meta[task].reindex(P.index); k = lab.notna()
            y, p = lab[k].astype(str).to_numpy(), P[k.to_numpy()].idxmax(axis=1).astype(str).to_numpy()
            seen = sorted(set(y) & eligible)
            f1 = f1_score(y, p, labels=seen, average="macro", zero_division=0)
            per = []
            for pl in plants:
                had = meta.loc[pl.index, task].notna().to_numpy()
                pl_t = pl[pl[task].notna() & pl[f"{task}_planted"].notna() & had]
                sc = F.flag_scores(P, pl_t[task]).dropna()
                per.append(pd.DataFrame({"score": sc.to_numpy(float), "planted": pl_t.loc[sc.index, f"{task}_planted"].astype(bool).to_numpy(),
                                         "project": meta.loc[sc.index, "archive_project"].astype(str).to_numpy()}))
            cs = np.nanmean([F.choosing_score(t.score.to_numpy(), t.planted.to_numpy())[0] for t in per])
            r5 = np.nanmean([F.recall_at_budget(t.score.to_numpy(), t.planted.to_numpy(), 0.05)[0] for t in per])
            codes = [pd.Categorical(t.project, categories=projects).codes for t in per]
            bcs = []
            for draw in draws:
                counts = np.bincount(draw, minlength=len(projects)); css = []
                for t, c in zip(per, codes):
                    idx = np.repeat(np.arange(len(t)), counts[c])
                    if len(idx): css.append(F.choosing_score(t.score.to_numpy()[idx], t.planted.to_numpy()[idx])[0])
                bcs.append(np.nanmean(css) if css and np.isfinite(css).any() else np.nan)
            boot[name] = np.asarray(bcs, float)
            rows.append({"task": task, "model": name, "f1_macro_eligible": round(f1, 3), "recall_at_5pct": round(float(r5), 3),
                         "choosing_score": round(float(cs), 3) if np.isfinite(cs) else np.nan})
        ref = "DIANA+LogReg"
        for name in models:
            if name == ref:
                continue
            d = boot[name] - boot[ref]; d = d[np.isfinite(d)]
            if not len(d):
                pairs.append({"task": task, "combo": name, "delta_choosing_vs_DIANA+LogReg": np.nan, "ci_low": np.nan, "ci_high": np.nan, "verdict": "can't check"}); continue
            lo, hi = np.percentile(d, [2.5, 97.5])
            obs = next(r["choosing_score"] for r in rows if r["task"] == task and r["model"] == name) - next(r["choosing_score"] for r in rows if r["task"] == task and r["model"] == ref)
            pairs.append({"task": task, "combo": name, "delta_choosing_vs_DIANA+LogReg": round(float(obs), 3) if np.isfinite(obs) else np.nan,
                          "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                          "verdict": "tie" if lo <= 0 <= hi else (name if obs > 0 else ref)})
    R, PR = pd.DataFrame(rows), pd.DataFrame(pairs)
    R.to_csv(OUT / "combo_scores.tsv", sep="\t", index=False); PR.to_csv(OUT / "combo_paired_vs_declared.tsv", sep="\t", index=False)
    pd.set_option("display.width", 240)
    print("dev-fold OOF, all models and averages:")
    print(R.pivot(index="model", columns="task", values="choosing_score").to_string())
    print()
    print(R.pivot(index="model", columns="task", values="f1_macro_eligible").to_string())
    print("\npaired against DIANA+LogReg (choosing score):")
    print(PR.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
