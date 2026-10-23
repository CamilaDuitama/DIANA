#!/usr/bin/env python3
"""Real-mislabel replay on the v19 held-out set (ledger read 4, recorded 2026-10-22 before scoring).

The genuine AncientMetagenomeDir corrections (results/amd_label_corrections/corrections.tsv,
change_type "genuine relabel") whose run is a v19 held-out run are replayed on the stored held-out
probabilities of DIANA (read 2) and of logistic regression and random forest (read 1); nothing is
refitted. Per correction and model:
  stated label  the superseded label; "cannot assess" if it is not a class the model was trained on
  flag score    1 - P(superseded label)
  flagged       at the 5 / 10 / 20 % budgets set by the correctly labelled held-out runs of that field
                (every other labelled held-out run, stated label = its current label), with the scoring
                module's recall_at_budget; NaN ("can't check") where the budget is unattainable
  prediction    the top class (DIANA: "no call" below its cutoff) and whether it is the corrected label
Descriptive (11 runs, 2 studies), never a rate. Writes results/heldout_v19/real_mislabels/replay.tsv.
The released DIANA (read 5): --diana results/heldout_v19/diana_refit/probs_DIANA_{task}.tsv
--out results/heldout_v19/real_mislabels_refit (scripts/analysis/run_refit_scoring_v19.sbatch).

    ./env/bin/python scripts/analysis/73_real_mislabel_replay_v19.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
OUT = ROOT / "results/heldout_v19/real_mislabels"
MODELS = {"DIANA": "results/heldout_v19/diana/probs_DIANA_{task}.tsv",
          "LogReg": "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv",
          "RandomForest": "results/heldout_v19/baselines/{task}/probs_RandomForest_Bal.tsv"}
CUTOFFS = json.load(open(ROOT / "results/posthoc_v19/abstention_v19.json"))["min_confidence"]


def corrections(test: pd.DataFrame) -> pd.DataFrame:
    c = pd.read_csv(ROOT / "results/amd_label_corrections/corrections.tsv", sep="\t", dtype=str)
    c = c[(c.change_type == "genuine relabel") & c.archive_data_accession.isin(set(test.index))]
    c = c.rename(columns={"archive_data_accession": "Run_accession"})
    current = [test.at[r, f] for r, f in zip(c.Run_accession, c.field)]
    if [str(x) for x in current] != list(c.new):
        raise ValueError("a corrected run's current label differs from the correction's new label")
    return c[["Run_accession", "field", "old", "new", "commit"]].reset_index(drop=True)


def replay_one(probs: pd.DataFrame, test: pd.DataFrame, field: str, run: str, old: str, new: str,
               corrected_runs: set, model: str) -> dict:
    pc = {c[2:]: c for c in probs.columns if c.startswith("p_")}
    row = {"stated_old_in_training": old in pc}
    p = probs.loc[run, list(pc.values())].astype(float)
    top, top_p = p.idxmax()[2:], float(p.max())
    row["prediction"] = "no call" if model == "DIANA" and top_p < CUTOFFS[field] else top
    row["prediction_is_corrected_label"] = row["prediction"] == new
    row["p_old"] = float(p[pc[old]]) if old in pc else np.nan
    row["p_new"] = float(p[pc[new]]) if new in pc else np.nan
    if old not in pc:
        row["verdict"] = "cannot assess"
        return row
    clean = test.loc[test[field].notna() & ~test.index.isin(corrected_runs), field]
    clean = clean[clean.isin(pc) & clean.index.isin(probs.index)]
    clean_score = 1.0 - np.array([float(probs.at[r, pc[lab]]) for r, lab in clean.items()])
    score = np.append(clean_score, 1.0 - row["p_old"])
    planted = np.append(np.zeros(len(clean_score), bool), True)
    att = S.attainable_budgets(score, planted)
    row["flag_score"] = 1.0 - row["p_old"]
    row["rank_among_clean"] = float((clean_score >= row["flag_score"]).mean())   # share of clean runs scoring as high
    for b in S.BUDGETS:
        row[f"flagged_{int(b * 100)}pct"] = S.recall_if_attainable(score, planted, b, att)
    row["verdict"] = "flagged at 5 %" if row["flagged_5pct"] == 1 else ("can't check at 5 %" if np.isnan(row["flagged_5pct"]) else "not flagged at 5 %")
    return row


def main() -> None:
    import argparse
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diana", default=None, help="DIANA probability template with {task} instead of the default")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--cutoffs", type=Path, default=None, help="no-call cutoffs JSON instead of abstention_v19.json")
    a = ap.parse_args()
    OUT = a.out
    if a.cutoffs:
        CUTOFFS.update(json.load(open(a.cutoffs))["min_confidence"])
    if a.diana:
        MODELS["DIANA"] = a.diana
    test = pd.read_csv(ROOT / "data/splits_v19/test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    c = corrections(test)
    rows = []
    for model, tmpl in MODELS.items():
        for field in sorted(c.field.unique()):
            probs = pd.read_csv(ROOT / tmpl.format(task=field), sep="\t").set_index("Run_accession")
            sub = c[c.field == field]
            for r in sub.itertuples(index=False):
                rows.append({"model": model, "Run_accession": r.Run_accession, "field": field, "old": r.old, "new": r.new,
                             "commit": r.commit, **replay_one(probs, test, field, r.Run_accession, r.old, r.new, set(c.Run_accession), model)})
    d = pd.DataFrame(rows)
    d = d.merge(test[["archive_project"]], left_on="Run_accession", right_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "replay.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    g = d.groupby(["model", "archive_project", "field", "old", "new"])
    summary = g.agg(runs=("Run_accession", "size"), flagged_5pct=("flagged_5pct", "sum"),
                    cant_check=("verdict", lambda x: int((x == "can't check at 5 %").sum())),
                    cannot_assess=("verdict", lambda x: int((x == "cannot assess").sum())), median_rank=("rank_among_clean", "median"),
                    predicts_corrected=("prediction_is_corrected_label", "sum"),
                    predictions=("prediction", lambda x: ", ".join(f"{k} {v}" for k, v in x.value_counts().items())))
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
