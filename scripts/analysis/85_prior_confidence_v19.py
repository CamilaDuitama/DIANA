#!/usr/bin/env python3
"""DIANA's reported confidence with the class prior added back (v19; PROTOCOLS.md amendments 2026-10-26, decision "b").

Labels stay the argmax of DIANA's probabilities; the reported confidence is P'(label), P' = P with the class prior added back
(network: P * pi^tau, tau from configs/v19_final/multitask.json; class-balanced logistic regression: P * pi; community type =
mean of the two; pi = class shares of the labelled training runs: training folds on the dev folds, all training runs for
held-out and release). The flag score stays 1 - P(stated) on the uncorrected probabilities.
  1. Dev folds: no-call cutoffs by the fixed-coverage rule on P'(label) (20th percentile over the scored dev runs) ->
     results/prior_v19/abstention_prior_coverage80.json, with the training class shares and tau for diana-predict.
  2. Held-out (re-score of the stored read-5 probabilities): per task confidence_<task>.tsv (label, confidence, original
     confidence), nocall_calibration_test.tsv and reliability_test.tsv in the layout of 49_ ->
     results/heldout_v19/diana_refit_prior/; the real-mislabel replay with DIANA's no calls recomputed ->
     results/heldout_v19/real_mislabels_refit_prior/replay.tsv.

    ./env/bin/python scripts/analysis/85_prior_confidence_v19.py
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
SPLIT = ROOT / "data/splits_v19"
COVERAGE = 0.80
HELD = ROOT / "results/heldout_v19/diana_refit"
HELD_LR = ROOT / "results/heldout_v19/baselines/{task}/probs_LogisticRegression_Bal.tsv"
OUT_DEV, OUT_HELD = ROOT / "results/prior_v19", ROOT / "results/heldout_v19/diana_refit_prior"
REPLAY_IN, REPLAY_OUT = ROOT / "results/heldout_v19/real_mislabels_refit_cov80", ROOT / "results/heldout_v19/real_mislabels_refit_prior"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/analysis/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probs(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", dtype={"Run_accession": str}).set_index("Run_accession")
    return d[[c for c in d.columns if c.startswith("p_")]].astype(float)


def corrected(P: pd.DataFrame, pi: pd.Series, tau: float) -> pd.DataFrame:
    X = P.to_numpy() * pi.reindex([c[2:] for c in P.columns]).fillna(0.0).to_numpy()[None, :] ** tau
    s = X.sum(axis=1, keepdims=True)
    return pd.DataFrame(np.where(s > 0, X / np.where(s > 0, s, 1), P.to_numpy()), index=P.index, columns=P.columns)


def label_and_conf(P: pd.DataFrame, Pc: pd.DataFrame) -> pd.DataFrame:
    Pc = Pc.reindex(index=P.index, columns=P.columns).fillna(0.0)
    lab = P.idxmax(axis=1)
    return pd.DataFrame({"label": lab.str[2:], "confidence": [Pc.at[r, c] for r, c in lab.items()],
                         "confidence_original": P.max(axis=1)})


def main() -> None:
    global HELD, OUT_DEV, OUT_HELD, REPLAY_IN, REPLAY_OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", action="store_true", help="DIANA without logistic regression (PROTOCOLS.md v19 amendment 2026-10-26)")
    net_only = ap.parse_args().net
    dev_dir = "results/devfold_probs_v19"
    if net_only:
        dev_dir, HELD = "results/devfold_probs_v19net", ROOT / "results/heldout_v19/diana_net"
        OUT_DEV, OUT_HELD = ROOT / "results/prior_v19net", ROOT / "results/heldout_v19/diana_net_prior"
        REPLAY_IN, REPLAY_OUT = ROOT / "results/heldout_v19/real_mislabels_net", ROOT / "results/heldout_v19/real_mislabels_net_prior"
    R = module("83_prior_correction_v19")
    tau = R.find_tau(R.TAU)
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    cut = {"rule": f"fixed coverage {COVERAGE:.0%} on the reported confidence P'(label), scored dev-fold runs (out-of-fold)",
           "coverage": COVERAGE, "tau_network": tau, "tau_logistic_regression": 1.0, "min_confidence": {}, "dev": {}, "class_prior": {}}
    OUT_HELD.mkdir(parents=True, exist_ok=True)
    summary, curves, conf_held = [], [], {}
    for task in TASKS:
        # 1. dev folds
        orig = probs(ROOT / f"{dev_dir}/probs_DIANA_{task}.tsv")
        corr = probs(OUT_DEV / f"probs_DIANAprior_{task}.tsv")
        sc = S.scored_devfold(tr[task], folds)
        runs = sc[sc].index.intersection(orig.index)
        lc = label_and_conf(orig.loc[runs], corr.loc[runs])
        conf = np.sort(lc.confidence.to_numpy())
        c = float(conf[int(np.floor((1 - COVERAGE) * len(conf)))])
        keep = lc.confidence >= c
        y = tr.loc[runs, task].astype(str)
        if keep.mean() < COVERAGE:
            raise ValueError(f"{task}: cutoff answers {keep.mean():.3f} < {COVERAGE}")
        pi = tr[task].dropna().astype(str).value_counts(normalize=True)
        cut["min_confidence"][task] = c
        cut["dev"][task] = {"runs": int(len(runs)), "answered": float(keep.mean()),
                            "accuracy_answered": float((lc.label[keep] == y[keep]).mean()), "accuracy_all": float((lc.label == y).mean())}
        cut["class_prior"][task] = {k: float(v) for k, v in pi.items()}
        # 2. held-out
        net = probs(HELD / f"probs_Shared_{task}.tsv")
        dia = probs(HELD / f"probs_DIANA_{task}.tsv")
        net_c = corrected(net, pi, tau)
        if task == "community_type" and not net_only:
            lr = probs(Path(str(HELD_LR).format(task=task))).reindex(index=net.index, columns=net.columns).fillna(0.0)
            dia_c = 0.5 * net_c + 0.5 * corrected(lr, pi, 1.0)
        else:
            dia_c = net_c
        lch = label_and_conf(dia, dia_c)
        lch.insert(0, "y_true", te.loc[lch.index, task])
        lch.reset_index().to_csv(OUT_HELD / f"confidence_{task}.tsv", sep="\t", index=False)
        conf_held[task] = lch
        sh = S.scored_heldout(te[task], set(tr[task].dropna()))
        d = lch.loc[sh[sh].index]
        correct = (d.label == d.y_true.astype(str)).to_numpy()
        top = d.confidence.to_numpy()
        keep = top >= c
        t, ece = module("49_diana_heldout_v19").reliability(top, correct)
        t.insert(0, "task", task)
        curves.append(t)
        summary.append({"task": task, "scored_runs": int(len(d)), "cutoff": c, "answered_share": float(keep.mean()),
                        "accuracy_answered": float(correct[keep].mean()), "accuracy_all": float(correct.mean()), "ece": ece})
        log.info("%s: dev cutoff %.4f (answers %.3f); held-out answers %.3f, accuracy answered %.3f, ECE %.3f", task, c,
                 cut["dev"][task]["answered"], keep.mean(), correct[keep].mean(), ece)
    json.dump(cut, open(OUT_DEV / "abstention_prior_coverage80.json", "w"), indent=1)
    pd.DataFrame(summary).to_csv(OUT_HELD / "nocall_calibration_test.tsv", sep="\t", index=False)
    pd.concat(curves).to_csv(OUT_HELD / "reliability_test.tsv", sep="\t", index=False)
    # 3. real-mislabel replay: DIANA's no calls with the new confidence and cutoffs (flags unchanged)
    rp = pd.read_csv(REPLAY_IN / "replay.tsv", sep="\t", dtype={"Run_accession": str})
    m = rp.model == "DIANA"
    for i in rp.index[m]:
        lc = conf_held[rp.at[i, "field"]].loc[rp.at[i, "Run_accession"]]
        pred = lc.label if lc.confidence >= cut["min_confidence"][rp.at[i, "field"]] else "no call"
        rp.at[i, "prediction"] = pred
        rp.at[i, "prediction_is_corrected_label"] = pred == rp.at[i, "new"]
    REPLAY_OUT.mkdir(parents=True, exist_ok=True)
    rp.to_csv(REPLAY_OUT / "replay.tsv", sep="\t", index=False)
    log.info("replay: DIANA predicts the corrected label for %d of %d, no call on %d", int(rp[m].prediction_is_corrected_label.sum()),
             int(m.sum()), int((rp[m].prediction == "no call").sum()))
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(json.dumps({k: cut[k] for k in ("min_confidence", "dev")}, indent=1))


if __name__ == "__main__":
    main()
