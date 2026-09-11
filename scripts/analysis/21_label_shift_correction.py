#!/usr/bin/env python3
"""E2: correct for label shift at inference, validated on the dev folds.

The problem, measured
---------------------
Train and held-out class distributions differ by total-variation distance 0.24 to
0.41, and simulating 1,573 alternative BioProject-disjoint splits of the same corpus
shows that shift is **intrinsic** to study-disjoint splitting rather than an artefact
of this split: even the best possible split has TVD 0.30 on `feature`. Studies
specialise in sample types, so any study-disjoint evaluation trains under one class
prior and tests under a different one.

The remedy
----------
Saerens-Latinne-Decaestecker EM. Iterate: reweight each posterior by the ratio of the
current prior estimate to the training prior, renormalise, then set the new prior
estimate to the mean reweighted posterior. It converges to a maximum-likelihood
estimate of the test prior using **only the unlabelled** inputs, and the reweighted
posteriors are the corrected predictions.

This is transductive: it needs the test *features*, though never the test labels. That
has to be disclosed. It is also the realistic deployment setting, since a curator has
the samples they want labelled in hand.

Validated here on the out-of-fold dev-fold probabilities from the calibration refits,
where each fold is BioProject-disjoint from the model that predicted it, so it
simulates the held-out situation without touching held-out.

    ./env/bin/python scripts/analysis/21_label_shift_correction.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
# The logit_adjust_tau each single-task model was trained with, from its search.
TAU = {"community_type": 0.644, "feature": 0.165, "sample_host": 0.080,
       "material": 0.142}
EPS = 1e-12


def load_fold(task: str, fold: int, glob_cls, gidx):
    """(probabilities in the global class space, true global index) for one dev fold."""
    base = PROJECT_ROOT / "results/calibration_v9"
    enc = json.loads((base / f"fit_{task}_fold{fold}/label_encoders.json").read_text())
    fold_cls = enc[task]["classes"] if isinstance(enc[task], dict) else enc[task]
    d = pd.read_csv(base / f"oof_{task}_fold{fold}/test_predictions.tsv", sep="\t")
    cols = sorted((c for c in d.columns if c.startswith(f"{task}_prob_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    d = d[d[f"{task}_true"].notna()]
    d = d[d[f"{task}_true"].astype(str).isin(gidx)]
    P = np.zeros((len(d), len(glob_cls)))
    src = d[cols].to_numpy(dtype=float)
    for j, c in enumerate(fold_cls):
        if str(c) in gidx:
            P[:, gidx[str(c)]] = src[:, j]
    y = d[f"{task}_true"].astype(str).map(gidx).to_numpy(dtype=int)
    return P, y


def effective_prior(prior_train: np.ndarray, tau: float) -> np.ndarray:
    """The prior the trained model's output actually reflects.

    Logit adjustment adds tau * log(prior) to the logits during TRAINING, so the
    fitted model already emits an approximately prior-corrected posterior, close to
    p(y|x) / prior^tau. SLD assumes the posterior reflects the RAW training prior;
    feeding it the raw prior therefore applies a second correction on top of the
    first, and the EM runs away. On the first attempt it collapsed every fold's prior
    estimate onto one class (TVD 1.000 from the truth) and F1 fell to ~0.

    The reference prior is prior^(1-tau), renormalised.
    """
    q = np.power(np.maximum(prior_train, EPS), 1.0 - tau)
    q[prior_train <= EPS] = 0.0
    return q / max(q.sum(), EPS)


def sld(P: np.ndarray, prior_ref: np.ndarray, n_iter: int = 100, tol: float = 1e-7,
        damping: float = 0.5):
    """Saerens-Latinne-Decaestecker EM, damped.

    Damping is not cosmetic here: undamped EM on these posteriors diverges, and a
    convex step between the old and new estimate is the standard remedy. Returns the
    corrected posteriors, the prior estimate, and whether it converged.
    """
    pi = prior_ref.copy()
    Pc = P.copy()
    converged = False
    for _ in range(n_iter):
        r = np.where(prior_ref > EPS, pi / np.maximum(prior_ref, EPS), 0.0)
        Pc = P * r
        Pc = np.divide(Pc, np.maximum(Pc.sum(axis=1, keepdims=True), EPS))
        new = damping * Pc.mean(axis=0) + (1.0 - damping) * pi
        if np.abs(new - pi).max() < tol:
            pi = new
            converged = True
            break
        pi = new
    return Pc, pi, converged


def bbse(P_src: np.ndarray, y_src: np.ndarray, P_tgt: np.ndarray, K: int):
    """Black Box Shift Estimation (Lipton et al. 2018), the stable alternative.

    SLD diverges on this corpus. Class priors span 1/2000 to 0.55, and its ratio
    pi / pi_ref explodes for a near-zero-prior class, amplifying that class and
    raising its own estimate: on `sample_host` fold 2 it put the entire prior on
    `Papio sp.`, one training run, true frequency zero.

    BBSE avoids the iteration entirely. Build the soft confusion matrix
    C[i, j] = mean predicted probability of class i over source rows truly of class j,
    take q = mean predicted distribution on the target, and solve C @ pi = q once.
    No feedback loop, so no runaway. Solved by least squares with the estimate
    projected onto the simplex, because C is ill-conditioned when classes are rare.
    """
    C = np.zeros((K, K))
    for j in range(K):
        m = y_src == j
        if m.any():
            C[:, j] = P_src[m].mean(axis=0)
    q = P_tgt.mean(axis=0)
    pi, *_ = np.linalg.lstsq(C, q, rcond=None)
    pi = np.clip(pi, 0.0, None)
    tot = pi.sum()
    return (pi / tot) if tot > EPS else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/label_shift_v9")
    args = ap.parse_args()

    md = pd.read_csv(PROJECT_ROOT / "data/splits_v9/train_metadata.tsv",
                     sep="\t", low_memory=False)
    el = pd.read_csv(PROJECT_ROOT / "data/splits_v9/class_eligibility.tsv", sep="\t")
    rows = []
    print("Dev-fold validation: f1_macro_eligible before and after SLD correction\n")
    print(f"{'task':<16}{'fold':>5}{'n':>6}{'before':>9}{'after':>9}{'change':>9}"
          f"{'prior TVD found':>17}")
    for task in TASKS:
        glob_cls = sorted(md[task].dropna().astype(str).unique())
        gidx = {c: i for i, c in enumerate(glob_cls)}
        elig = set(el[(el.target == task) & el.evaluable]["class"].astype(str))
        for fold in range(5):
            P, y = load_fold(task, fold, glob_cls, gidx)
            if len(y) < 20:
                continue
            # the prior the model was trained under: this fold's own training portion
            fit_ids = set((PROJECT_ROOT /
                f"results/calibration_v9/ids/cal_train_fold{fold}.txt").read_text().split())
            sub = md[md.Run_accession.astype(str).isin(fit_ids)]
            cnt = sub[task].astype(str).value_counts()
            prior = np.array([cnt.get(c, 0) for c in glob_cls], dtype=float)
            prior = prior / max(prior.sum(), EPS)

            tau = TAU[task]
            ref = effective_prior(prior, tau)
            Pc, pi, conv = sld(P, ref)
            def f1(M):
                pred = np.array(glob_cls)[M.argmax(1)]
                yt = np.array(glob_cls)[y]
                keep = sorted(c for c in set(yt) if c in elig)
                return f1_score(yt, pred, labels=keep, average="macro",
                                zero_division=0) if keep else np.nan
            b, a = f1(P), f1(Pc)
            true_pi = np.array([(y == i).mean() for i in range(len(glob_cls))])
            rows.append({"task": task, "fold": fold, "n": int(len(y)),
                         "f1_before": b, "f1_after": a, "delta": a - b,
                         "tvd_prior_estimated_vs_true":
                             0.5 * float(np.abs(pi - true_pi).sum()),
                         "tvd_train_vs_true": 0.5 * float(np.abs(prior - true_pi).sum()),
                         "tvd_reference_vs_true": 0.5 * float(np.abs(ref - true_pi).sum()),
                         "converged": bool(conv), "tau": tau})
            print(f"{task:<16}{fold:>5}{len(y):>6}{b:>9.3f}{a:>9.3f}{a-b:>+9.3f}"
                  f"{rows[-1]['tvd_prior_estimated_vs_true']:>17.3f}")
        print()
    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "dev_fold_sld.tsv", sep="\t", index=False)
    print(f"{'task':<16}{'mean before':>13}{'mean after':>12}{'mean change':>13}"
          f"{'folds improved':>16}")
    for task in TASKS:
        g = df[df.task == task]
        if g.empty:
            continue
        print(f"{task:<16}{g.f1_before.mean():>13.3f}{g.f1_after.mean():>12.3f}"
              f"{g.delta.mean():>+13.3f}{f'{int((g.delta>0).sum())}/{len(g)}':>16}")
    print(f"\nwrote {args.output / 'dev_fold_sld.tsv'}")
    print("\n'prior TVD found' is how far the EM estimate lands from the fold's TRUE")
    print("prior. Compare it to tvd_train_vs_true in the file: if EM is not closer,")
    print("the estimate is not working and any F1 change is incidental.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
