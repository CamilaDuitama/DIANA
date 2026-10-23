#!/usr/bin/env python3
"""v19 similarity baselines on the dev folds (PROTOCOLS.md amendment of 2026-10-15, "A and B").

Similarity of two runs = exact Jaccard of their sketches (k = 31, 1 in 1,000), from
results/pairwise_similarity_v19/similarity.npz.

  vote    (A) similarity vote: the k most similar labelled runs of the other folds vote; grid
          k x weight (uniform / similarity) x one-per-BioProject; class probability = weighted
          vote share; tie in the label -> larger total similarity, then alphabetical. Also reports
          the published flag rule of Faure et al. 2026 on the dev plantings (faure_rule.tsv).
  profile (B) similarity profile: a run is described by its similarity to every run of the
          fold's training part, with same-BioProject entries set to 0; one standard baseline
          with its v19 grid.

Both select by the v19 rule (dev-fold macro-F1 on scored runs, first in grid order on a tie) and
write the 40_baselines_v19.py layout: grid_<model>.tsv, best_<model>.json, oof_pred_<model>.tsv,
oof_probs_<model>.tsv.

    ./env/bin/python scripts/evaluation/45_similarity_baselines_v19.py vote --task feature
    ./env/bin/python scripts/evaluation/45_similarity_baselines_v19.py profile --task feature --model kNN
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

SPLIT = ROOT / "data/splits_v19"
SIM = ROOT / "results/pairwise_similarity_v19/similarity.npz"
OUT_VOTE = ROOT / "results/baselines_sim_v19"
OUT_PROFILE = ROOT / "results/baselines_simprofile_v19"
PLANTINGS = ROOT / "results/planted_v19/dev"
VOTE_GRID = [{"k": k, "weight": w, "one_per_project": o}
             for k, w, o in product((1, 3, 5, 10, 20), ("uniform", "similarity"), (False, True))]
FAURE = {"min_similarity": 0.1, "min_neighbours": 4, "min_agreement": 0.8}
TIE_EPS = 1e-6

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(task: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Training metadata (fold, label, BioProject) and the similarity matrix in the same row order."""
    meta = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    folds = pd.read_csv(SPLIT / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold
    meta["fold"] = meta.Run_accession.map(folds)
    if meta.fold.isna().any():
        raise SystemExit("training runs without a dev fold")
    meta["fold"] = meta.fold.astype(int)
    z = np.load(SIM, allow_pickle=True)  # our own file (run ids saved as objects)
    pos = pd.Series(np.arange(len(z["runs"])), index=z["runs"].astype(str))
    missing = set(meta.Run_accession) - set(pos.index)
    if missing:
        raise SystemExit(f"{len(missing)} training runs have no similarity row")
    idx = pos.loc[meta.Run_accession].to_numpy()
    J = z["exact"][np.ix_(idx, idx)].astype(np.float64)
    return meta, J


def neighbours(sim: np.ndarray, proj: np.ndarray, one_per_project: bool) -> np.ndarray:
    """Candidate positions ranked by similarity (stable), optionally the best one per BioProject."""
    order = np.argsort(-sim, kind="stable")
    if one_per_project:
        _, first = np.unique(proj[order], return_index=True)
        order = order[np.sort(first)]
    return order


def vote(sim: np.ndarray, labels: np.ndarray, order: np.ndarray, k: int, weight: str,
         classes: list[str]) -> tuple[np.ndarray, str]:
    """Class probabilities (weighted vote shares over `classes`) and the predicted label."""
    top = order[:k]
    w = sim[top] if weight == "similarity" else np.ones(len(top))
    if w.sum() <= 0:  # every voter has similarity 0: uniform vote (logged by run_vote)
        w = np.ones(len(top))
    share = {c: 0.0 for c in classes}
    total = {c: 0.0 for c in classes}
    for lab, wi, si in zip(labels[top], w, sim[top]):
        share[lab] += wi
        total[lab] += si
    p = np.array([share[c] for c in classes]) / w.sum()
    pred = sorted(classes, key=lambda c: (-share[c], -total[c], c))[0]
    # the probabilities must carry the tie-break too: scorers take their argmax, which on a tie
    # would pick the alphabetically first class instead of `pred`
    p[classes.index(pred)] += TIE_EPS
    return p / p.sum(), pred


def write(out: Path, name: str, meta: pd.DataFrame, task: str, classes: list[str], rows: list[dict],
          best: tuple, scored: np.ndarray) -> None:
    score, i, params, pred, prob = best
    pd.DataFrame(rows).to_csv(out / f"grid_{name}.tsv", sep="\t", index=False)
    lab = meta[task].notna().to_numpy()
    base = pd.DataFrame({"Run_accession": meta.Run_accession, "fold": meta.fold, "y_true": meta[task],
                         "scored": scored, "y_pred": pred})
    base[lab].to_csv(out / f"oof_pred_{name}.tsv", sep="\t", index=False)
    if prob is not None:
        pd.concat([base[lab].reset_index(drop=True), pd.DataFrame(prob[lab], columns=[f"p_{c}" for c in classes])],
                  axis=1).to_csv(out / f"oof_probs_{name}.tsv", sep="\t", index=False)
    json.dump({"task": task, "model": name, "config": i, "params": params, "f1_macro_eligible": score,
               "n_scored": int(scored.sum()), "rule": "PROTOCOLS.md v19 C2e; amendment 2026-10-15 (A and B)"},
              open(out / f"best_{name}.json", "w"), indent=1)
    log.info("%s / %s: selected %s (%.4f)", task, name, params, score)


def setup(task: str, out_root: Path, name: str):
    out = out_root / task
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"best_{name}.json").exists():
        raise SystemExit(f"{out}/best_{name}.json exists; refusing to overwrite")
    meta, J = load(task)
    labels = pd.Series(meta[task].to_numpy(), index=meta.Run_accession.to_numpy())
    scored = S.scored_devfold(labels, pd.Series(meta.fold.to_numpy(), index=labels.index)).to_numpy()
    eligible = S.eligible_classes(SPLIT / "class_eligibility.tsv", task)
    classes = sorted(meta[task].dropna().unique())
    return out, meta, J, scored, eligible, classes


def run_vote(task: str) -> None:
    out, meta, J, scored, eligible, classes = setup(task, OUT_VOTE, "SimVote")
    y = meta[task].astype(str).to_numpy()
    lab = meta[task].notna().to_numpy()
    proj = meta.archive_project.astype(str).to_numpy()
    fold = meta.fold.to_numpy()
    orders = {}
    for i in np.flatnonzero(lab):
        cand = np.flatnonzero(lab & (fold != fold[i]))
        orders[i] = (cand, {o: neighbours(J[i, cand], proj[cand], o) for o in (False, True)})
    zero = sum(J[i, cand][ords[False][0]] <= 0 for i, (cand, ords) in orders.items() if len(cand))
    log.info("%s: %d labelled runs whose most similar candidate has similarity 0 (their vote is uniform)", task, zero)
    rows, best = [], None
    for gi, params in enumerate(VOTE_GRID):
        pred = np.full(len(meta), "", dtype=object)
        prob = np.zeros((len(meta), len(classes)))
        for i, (cand, ords) in orders.items():
            prob[i], pred[i] = vote(J[i, cand], y[cand], ords[params["one_per_project"]], params["k"], params["weight"], classes)
        score = S.f1_macro_eligible(y[scored], pred[scored].astype(str), eligible)
        if not np.isfinite(score):
            raise ValueError(f"SimVote {params}: non-finite score")
        rows.append({"config": gi, "params": json.dumps(params), "f1_macro_eligible": score})
        log.info("SimVote %s: %.4f", params, score)
        if best is None or score > best[0]:
            best = (score, gi, params, pred, prob.astype(np.float32))
    write(out, "SimVote", meta, task, classes, rows, best, scored)
    faure_rule(task, meta, J, out)


def faure_rule(task: str, meta: pd.DataFrame, J: np.ndarray, out: Path) -> None:
    """Their flag rule as one fixed operating point on each dev planting's pool (41_ pool rule)."""
    F = module(ROOT / "scripts/analysis/41_flagging_v19.py", "flag41")
    tr = meta.set_index("Run_accession")
    folds = tr.fold
    byfold = S.fit_classes_by_fold(tr[task], folds)
    assessable = lambda r, c: c in byfold[int(folds[r])]  # noqa: E731
    files = sorted(glob.glob(str(PLANTINGS / "planted_train_mixed_r0.1_s*.tsv")))
    plantings = F.read_plantings(files, SPLIT, tr.index)
    pos = pd.Series(np.arange(len(meta)), index=meta.Run_accession.to_numpy())
    y = meta[task].to_numpy(); proj = meta.archive_project.astype(str).to_numpy(); fold = meta.fold.to_numpy()
    lab = meta[task].notna().to_numpy()
    call = {}
    for r in tr.index[tr[task].notna()]:
        i = pos[r]
        cand = np.flatnonzero(lab & (fold != fold[i]))
        order = neighbours(J[i, cand], proj[cand], True)
        keep = order[J[i, cand][order] >= FAURE["min_similarity"]]
        if len(keep) < FAURE["min_neighbours"]:
            call[r] = None
            continue
        labs, counts = np.unique(y[cand][keep], return_counts=True)
        call[r] = labs[counts.argmax()] if counts.max() / len(keep) >= FAURE["min_agreement"] else ""
    rows = []
    for f, pl in zip(files, plantings):
        t = F.pool(tr, pl, task, assessable)[0]
        c = t.index.map(call)
        ok = pd.Series([v is not None for v in c], index=t.index)
        flag = pd.Series([bool(v) and v != s for v, s in zip(c, t.stated)], index=t.index)
        pl_ = t.planted.astype(bool)
        rows.append({"planting": Path(f).stem, "pool": len(t), "assessable": int(ok.sum()),
                     "planted": int(pl_.sum()), "planted_assessable": int((pl_ & ok).sum()),
                     "catch_rate": float(flag[pl_].mean()), "false_flag_rate": float(flag[~pl_].mean())})
    d = pd.DataFrame(rows)
    d.to_csv(out / "faure_rule.tsv", sep="\t", index=False)
    log.info("%s Faure rule: assessable %.3f of pool, catch %.3f, false flags %.3f (means over %d plantings)", task,
             (d.assessable / d["pool"]).mean(), d.catch_rate.mean(), d.false_flag_rate.mean(), len(d))


def run_profile(task: str, model: str, n_jobs: int) -> None:
    T = module(ROOT / "scripts/evaluation/40_baselines_v19.py", "base40")
    if model not in T.GRIDS:
        raise SystemExit(f"{model} is not a v19 baseline")
    out, meta, J, scored, eligible, classes = setup(task, OUT_PROFILE, model)
    y = meta[task].astype(str).to_numpy()
    lab = meta[task].notna().to_numpy()
    proj = meta.archive_project.astype(str).to_numpy()
    fold = meta.fold.to_numpy()
    Jz = np.where(proj[:, None] == proj[None, :], 0.0, J).astype(np.float32)
    rows, best = [], None
    for gi, params in enumerate(T.GRIDS[model]):
        pred = np.full(len(meta), "", dtype=object)
        prob = None if model in T.NO_PROBA else np.zeros((len(meta), len(classes)), dtype=np.float32)
        for k in sorted(set(fold)):
            cols = np.flatnonzero(fold != k)
            fit, ev = lab & (fold != k), lab & (fold == k)
            m = T.build(model, params, n_jobs).fit(Jz[np.ix_(fit, cols)], y[fit])
            X_ev = Jz[np.ix_(ev, cols)]
            pred[ev] = m.predict(X_ev)
            if prob is not None:
                p = m.predict_proba(X_ev)
                for j, c in enumerate(m.classes_):
                    prob[np.flatnonzero(ev), classes.index(c)] = p[:, j]
        score = S.f1_macro_eligible(y[scored], pred[scored].astype(str), eligible)
        if not np.isfinite(score):
            raise ValueError(f"{model} {params}: non-finite score")
        rows.append({"config": gi, "params": json.dumps(params), "f1_macro_eligible": score})
        log.info("%s profile %s: %.4f", model, params, score)
        if best is None or score > best[0]:
            best = (score, gi, params, pred, prob)
    write(out, model, meta, task, classes, rows, best, scored)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["vote", "profile"])
    ap.add_argument("--task", required=True, choices=["community_type", "feature", "sample_host", "material"])
    ap.add_argument("--model")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "vote":
        run_vote(a.task)
    else:
        if not a.model:
            raise SystemExit("--model is required for profile")
        run_profile(a.task, a.model, a.n_jobs)


if __name__ == "__main__":
    main()
