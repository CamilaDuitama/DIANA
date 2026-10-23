#!/usr/bin/env python3
"""Combination-form sweep (PROTOCOLS.md, 2026-10-05): weighted arithmetic averages of the
DIANA and logistic-regression out-of-fold probabilities, plus the geometric mean, each paired
against the declared fifty-fifty average on the dev folds.

    ./env/bin/python scripts/analysis/67_combination_form_sweep.py
"""
from __future__ import annotations

import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
F = __import__("59_devfold_flagging_score")
from diana.evaluation.metrics import load_eligible_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
OUT = ROOT / "results/combination_form_v9"
N_BOOT = 1000
WEIGHTS = (0.25, 0.4, 0.5, 0.6, 0.75)


def rng_for(*parts):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    plants = [pd.read_csv(f, sep="\t").set_index("Run_accession")
              for f in sorted(glob.glob(str(ROOT / "results/planted_mislabels_dev_v9/planted_train_mixed_r0.1_s*.tsv")))]
    rows, pairs = [], []
    for task in TASKS:
        D = F.reference_probs(task, ROOT / ("results/step4_2_studyweight_v9" if task == "feature" else "results/reference_v9"))
        L = F.baseline_probs("LogisticRegression_Bal", task)
        cols = sorted(set(D.columns) | set(L.columns)); idx = D.index.intersection(L.index)
        Dm = D.reindex(index=idx, columns=cols).fillna(0.0); Lm = L.reindex(index=idx, columns=cols).fillna(0.0)
        models = {f"arith_w{w:g}": w * Dm + (1 - w) * Lm for w in WEIGHTS}
        G = np.sqrt(np.clip(Dm.to_numpy(), 1e-9, 1) * np.clip(Lm.to_numpy(), 1e-9, 1))
        models["geometric_w0.5"] = pd.DataFrame(G / G.sum(axis=1, keepdims=True), index=idx, columns=cols)
        eligible = load_eligible_classes(ROOT / "data/splits_v9/class_eligibility.tsv", task)
        projects = np.array(sorted(meta.loc[meta[task].notna(), "archive_project"].astype(str).unique()))
        draws = [rng_for("form", task, i).integers(0, len(projects), size=len(projects)) for i in range(N_BOOT)]
        boot = {}
        for name, P in models.items():
            lab = meta[task].reindex(P.index); k = lab.notna()
            y, p = lab[k].astype(str).to_numpy(), P[k.to_numpy()].idxmax(axis=1).astype(str).to_numpy()
            f1 = f1_score(y, p, labels=sorted(set(y) & eligible), average="macro", zero_division=0)
            per = []
            for pl in plants:
                had = meta.loc[pl.index, task].notna().to_numpy()
                pl_t = pl[pl[task].notna() & pl[f"{task}_planted"].notna() & had]
                sc = F.flag_scores(P, pl_t[task]).dropna()
                per.append(pd.DataFrame({"score": sc.to_numpy(float), "planted": pl_t.loc[sc.index, f"{task}_planted"].astype(bool).to_numpy(),
                                         "project": meta.loc[sc.index, "archive_project"].astype(str).to_numpy()}))
            cs = np.nanmean([F.choosing_score(t.score.to_numpy(), t.planted.to_numpy())[0] for t in per])
            codes = [pd.Categorical(t.project, categories=projects).codes for t in per]
            bcs = []
            for draw in draws:
                counts = np.bincount(draw, minlength=len(projects)); css = []
                for t, c in zip(per, codes):
                    i2 = np.repeat(np.arange(len(t)), counts[c])
                    if len(i2): css.append(F.choosing_score(t.score.to_numpy()[i2], t.planted.to_numpy()[i2])[0])
                bcs.append(np.nanmean(css) if css and np.isfinite(css).any() else np.nan)
            boot[name] = np.asarray(bcs, float)
            rows.append({"task": task, "form": name, "f1_macro_eligible": round(float(f1), 3),
                         "choosing_score": round(float(cs), 3) if np.isfinite(cs) else np.nan})
        ref = "arith_w0.5"
        for name in models:
            if name == ref: continue
            d = boot[name] - boot[ref]; d = d[np.isfinite(d)]
            obs = next(r["choosing_score"] for r in rows if r["task"] == task and r["form"] == name) - next(r["choosing_score"] for r in rows if r["task"] == task and r["form"] == ref)
            lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
            pairs.append({"task": task, "form": name, "delta_choosing_vs_5050": round(float(obs), 3) if np.isfinite(obs) else np.nan,
                          "ci_low": round(float(lo), 3), "ci_high": round(float(hi), 3),
                          "verdict": "can't check" if not np.isfinite(lo) or not np.isfinite(obs) else "tie" if lo <= 0 <= hi else (name if obs > 0 else ref)})
    R, PR = pd.DataFrame(rows), pd.DataFrame(pairs)
    R.to_csv(OUT / "form_scores.tsv", sep="\t", index=False); PR.to_csv(OUT / "form_paired.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(R.pivot(index="form", columns="task", values="choosing_score").to_string())
    print(); print(R.pivot(index="form", columns="task", values="f1_macro_eligible").to_string())
    print(); print(PR[PR.verdict != "tie"].to_string(index=False)); print("(rows not shown are ties)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
