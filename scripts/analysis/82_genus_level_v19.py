#!/usr/bin/env python3
"""Sample host scored at genus level beside species level, on the held-out set (v19; descriptive re-score).

Stored held-out predictions only (released DIANA, read 5; baselines, read 1); nothing is refitted. Genus = first word of
the label (Pan troglodytes ellioti -> Pan). At genus level the classes are the genera of the eligible species classes,
grouped into many/medium/few by their own training-run counts (as species classes are). Per level, whole field and by
group: balanced accuracy and macro-F1 with 95 % intervals over held-out BioProjects, and DIANA minus each baseline
(paired; intervals from the same resamples, p from the BioProject permutation test). Species-level keys are those of 75_ so its values are reproduced. Writes
results/genus_level_v19/{scores,paired}.tsv and the grid results/paper_v19/sup_genus_grid.png (Benjamini-Hochberg over
its cells; rows with fewer than 5 held-out BioProjects hatched and not tested, as in the main grids).

    ./env/bin/python scripts/analysis/82_genus_level_v19.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASK = "sample_host"
SPLIT = ROOT / "data/splits_v19"
OUT = ROOT / "results/genus_level_v19"
MODELS = {"DIANA": "results/heldout_v19/diana_net/probs_DIANA_{t}.tsv",
          "LogReg": "results/heldout_v19/baselines/{t}/probs_LogisticRegression_Bal.tsv",
          "RandomForest": "results/heldout_v19/baselines/{t}/probs_RandomForest_Bal.tsv",
          "kNN": "results/heldout_v19/baselines/{t}/probs_kNN.tsv",
          "Majority": "results/heldout_v19/baselines/{t}/probs_MajorityClass.tsv"}
METRICS = {"balanced_accuracy": S.balanced_accuracy_eligible, "macro_f1": S.f1_macro_eligible}


def genus(a) -> np.ndarray:
    return np.array([str(x).split()[0] for x in a])


def argmax(path: str, runs: pd.Index) -> np.ndarray:
    d = pd.read_csv(ROOT / path.format(t=TASK), sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    return d.loc[runs, [c for c in d.columns if c.startswith("p_")]].idxmax(axis=1).str[2:].to_numpy()


def grid_module():
    spec = importlib.util.spec_from_file_location("grid", ROOT / "scripts/paper/generate_results_figures_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    el = S.eligible_classes(SPLIT / "class_eligibility.tsv", TASK)
    sc = S.scored_heldout(te[TASK], set(tr[TASK].dropna()))
    runs = sc[sc].index
    g = te.loc[runs, "archive_project"].astype(str).to_numpy()
    y_sp = te.loc[runs, TASK].astype(str).to_numpy()
    pred = {m: argmax(p, runs) for m, p in MODELS.items()}
    tr_gen = tr[TASK].dropna().astype(str).map(lambda x: x.split()[0])
    levels = {"species": (y_sp, el, tr[TASK], lambda a: a, "species"),
              "genus": (genus(y_sp), {c.split()[0] for c in el}, tr_gen, genus, "genus")}
    scores, pairs, studies = [], [], {}
    for lev, (y, eligible, train_labels, f, tag) in levels.items():
        scopes = [("all", eligible & set(y), f"cls|{TASK}" if lev == "species" else f"genus|{TASK}",
                   f"pair|{TASK}" if lev == "species" else f"genuspair|{TASK}")]
        for reg, cls in S.regime_classes(train_labels, eligible).items():
            if cls & set(y):
                scopes.append((reg, cls & set(y), f"cls|{TASK}|{reg}" if lev == "species" else f"genus|{TASK}|{reg}",
                               f"regpair|{TASK}|{reg}" if lev == "species" else f"genuspair|{TASK}|{reg}"))
        for reg, cls, key, pkey in scopes:
            studies[(f"{TASK}_{tag}", reg)] = int(pd.Series(g)[np.isin(y, list(cls))].nunique())
            for metric, fn in METRICS.items():
                for m in MODELS:
                    r = S.score_with_interval(y, f(pred[m]), g, cls, key, metric=fn)
                    scores.append({"level": lev, "regime": reg, "metric": metric, "model": m, "value": r["value"],
                                   "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n_classes": len(cls)})
                for m in [x for x in MODELS if x != "DIANA"]:
                    r = S.paired_difference(runs, runs, y, f(pred["DIANA"]), f(pred[m]), g, cls, pkey, metric=fn)
                    pperm = S.paired_permutation_p(y, f(pred["DIANA"]), f(pred[m]), g, cls, f"perm|{pkey}|{metric}|{m}", metric=fn)
                    pairs.append({"task": f"{TASK}_{tag}", "level": lev, "regime": reg, "metric": metric, "a": "DIANA", "b": m,
                                  "delta": r["delta"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_boot": r["p_boot"],
                                  "p": pperm})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(OUT / "scores.tsv", sep="\t", index=False)
    p = pd.DataFrame(pairs)
    p.to_csv(OUT / "paired.tsv", sep="\t", index=False)
    G = grid_module()
    G.FIELDS = {f"{TASK}_species": "Sample host, species", f"{TASK}_genus": "Sample host, genus"}
    cols = [(met, b, f"{lab}<br>vs {short}") for met, lab in (("macro_f1", "Macro-F1"), ("balanced_accuracy", "Balanced accuracy"))
            for b, short in (("LogReg", "logistic<br>regression"), ("RandomForest", "random<br>forest"), ("kNN", "k-NN"))]
    G.winner_grid(p[p.regime != "all"], cols, ROOT / "results/paper_v19/sup_genus_grid.png", studies)
    pd.set_option("display.width", 220)
    s = pd.DataFrame(scores)
    s["cell"] = s.apply(lambda r: f"{r.value:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]", axis=1)
    print(s.pivot_table(index=["metric", "regime", "model"], columns="level", values="cell", aggfunc="first").to_string())
    p["cell"] = p.apply(lambda r: f"{r.delta:+.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}] p={r.p:.3f}", axis=1)
    print(p.pivot_table(index=["metric", "regime", "b"], columns="level", values="cell", aggfunc="first").to_string())
    print({k: v for k, v in studies.items()})


if __name__ == "__main__":
    main()
