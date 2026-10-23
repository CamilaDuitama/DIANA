#!/usr/bin/env python3
"""The v17 split optimisation, as run on 2026-10-07 (PROTOCOLS.md v17; v19 C8).

Recovered verbatim from the session record, where it ran as two inline commands and
was never saved; only file paths, logging and the solver report were added.

Step 1, held-out set. Binary x_p per BioProject (1 = held out) over the v9 corpus
(train + test, labels verified identical to AMD v26.03.0). For every eligible class k
(>= 2 BioProjects) with run counts C_kp, minimise the mean over k of
|sum_p C_kp x_p / C_k - 0.25|, subject to: held-out runs 23-27 % of all runs; at least
30 BioProjects on each side; every class with >= 2 BioProjects on both sides.
HiGHS through scipy.optimize.milp, relative gap 0. The minimax variant (minimise the
largest class deviation) is solved too and reported only.

Step 2, dev folds. On the resulting training side, binary x_pf (project p in fold f,
f = 0..4) minimising the mean over classes (>= 2 training BioProjects) and folds of
|share of the class's runs in fold f - 0.2|, subject to: one fold per project; each
fold 17-23 % of the training runs; every such class spanning at least two folds.
Relative gap 0.01, time limit 900 s, as run.

    ./env/bin/python scripts/data_prep/08_optimise_split_v17.py --out <dir>
then compare <dir> with results/split_optim_v17/.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/splits_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
TARGET, F = 0.25, 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def report(res, what: str) -> None:
    gap = getattr(res, "mip_gap", float("nan"))
    log.info("%s: status %s, mip gap %.4g, objective %.6f | %s", what, res.status, gap, res.fun, res.message[:80])


def heldout(out: Path) -> set:
    tr = pd.read_csv(SRC / "train_metadata.tsv", sep="\t", dtype=str).fillna("")
    te = pd.read_csv(SRC / "test_metadata.tsv", sep="\t", dtype=str).fillna("")
    d = pd.concat([tr, te], ignore_index=True)
    elig = pd.read_csv(SRC / "class_eligibility.tsv", sep="\t")
    elig_set = [(r.target, r["class"]) for _, r in elig.iterrows() if r.evaluable]
    projects = sorted(d.archive_project.unique())
    P = len(projects)
    n_p = d.archive_project.value_counts().reindex(projects).to_numpy(float)
    N = n_p.sum()
    C, names = [], []
    for t, k in elig_set:
        v = d[d[t] == k].archive_project.value_counts().reindex(projects).fillna(0).to_numpy(float)
        if v.sum() > 0:
            C.append(v)
            names.append((t, k))
    C = np.array(C)
    K = len(names)

    def constraints():
        cons = [LinearConstraint(np.hstack([C, -np.eye(K)]), -np.inf, TARGET * C.sum(1)),
                LinearConstraint(np.hstack([-C, -np.eye(K)]), -np.inf, -TARGET * C.sum(1)),
                LinearConstraint(np.concatenate([n_p, np.zeros(K)]), 0.23 * N, 0.27 * N),
                LinearConstraint(np.concatenate([np.ones(P), np.zeros(K)]), 30, P - 30)]
        rows, lo, hi = [], [], []
        for j in range(K):
            ind = (C[j] > 0).astype(float)
            if ind.sum() >= 2:
                rows.append(np.concatenate([ind, np.zeros(K)]))
                lo.append(1)
                hi.append(ind.sum() - 1)
        cons.append(LinearConstraint(np.array(rows), lo, hi))
        return cons

    nv = P + K
    cons = constraints()
    integ = np.concatenate([np.ones(P), np.zeros(K)])
    res1 = milp(c=np.concatenate([np.zeros(P), 1.0 / C.sum(1)]), constraints=cons, integrality=integ,
                bounds=Bounds(np.zeros(nv), np.concatenate([np.ones(P), np.full(K, np.inf)])),
                options={"time_limit": 600, "mip_rel_gap": 0.0})
    report(res1, "held-out, mean objective")
    x1 = np.round(res1.x[:P]).astype(bool)

    cons2 = [LinearConstraint(np.hstack([c.A, np.zeros((c.A.shape[0], 1))]), c.lb, c.ub) for c in cons]
    cons2.append(LinearConstraint(np.hstack([np.zeros((K, P)), np.diag(1.0 / C.sum(1)), -np.ones((K, 1))]),
                                  -np.inf, np.zeros(K)))
    res2 = milp(c=np.concatenate([np.zeros(nv), [1.0]]), constraints=cons2,
                integrality=np.concatenate([np.ones(P), np.zeros(K + 1)]),
                bounds=Bounds(np.zeros(nv + 1), np.concatenate([np.ones(P), np.full(K + 1, np.inf)])),
                options={"time_limit": 600, "mip_rel_gap": 0.0})
    report(res2, "held-out, minimax objective (reported only)")
    x2 = np.round(res2.x[:P]).astype(bool)

    out.mkdir(parents=True, exist_ok=True)
    pd.Series(np.array(projects)[x1]).to_csv(out / "optimal_mean_heldout_projects.txt", index=False, header=False)
    pd.Series(np.array(projects)[x2]).to_csv(out / "optimal_minimax_heldout_projects.txt", index=False, header=False)
    devs = np.array([abs(C[j][x1].sum() / C[j].sum() - TARGET) for j in range(K)])
    log.info("held-out: %d runs / %d projects, mean class deviation %.3f, max %.3f",
             int(n_p[x1].sum()), int(x1.sum()), devs.mean(), devs.max())
    return set(np.array(projects)[x1])


def dev_folds(hold: set, out: Path) -> None:
    d = pd.concat([pd.read_csv(SRC / f"{s}_metadata.tsv", sep="\t", dtype=str).fillna("")
                   for s in ["train", "test"]], ignore_index=True)
    tr = d[~d.archive_project.isin(hold)].reset_index(drop=True)
    projects = sorted(tr.archive_project.unique())
    P = len(projects)
    n_p = tr.archive_project.value_counts().reindex(projects).to_numpy(float)
    N = n_p.sum()
    C = []
    for t in TASKS:
        for k, g in tr[tr[t] != ""].groupby(t):
            if g.archive_project.nunique() >= 2:
                C.append(tr.assign(hit=(tr[t] == k)).groupby("archive_project").hit.sum()
                         .reindex(projects).fillna(0).to_numpy(float))
    C = np.array(C)
    K = len(C)
    nx, nd = P * F, K * F

    def xi(p, f):
        return p * F + f

    def di(k, f):
        return nx + k * F + f

    c = np.zeros(nx + nd)
    for k in range(K):
        for f in range(F):
            c[di(k, f)] = 1.0 / (F * C[k].sum())
    cons = []
    A = np.zeros((P, nx + nd))
    for p in range(P):
        for f in range(F):
            A[p, xi(p, f)] = 1
    cons.append(LinearConstraint(A, 1, 1))
    A = np.zeros((F, nx + nd))
    for f in range(F):
        for p in range(P):
            A[f, xi(p, f)] = n_p[p]
    cons.append(LinearConstraint(A, 0.17 * N, 0.23 * N))
    rows1, rows2, rhs = np.zeros((K * F, nx + nd)), np.zeros((K * F, nx + nd)), np.zeros(K * F)
    i = 0
    for k in range(K):
        for f in range(F):
            for p in range(P):
                rows1[i, xi(p, f)] = C[k, p]
                rows2[i, xi(p, f)] = -C[k, p]
            rows1[i, di(k, f)] = -1
            rows2[i, di(k, f)] = -1
            rhs[i] = (1.0 / F) * C[k].sum()
            i += 1
    cons.append(LinearConstraint(rows1, -np.inf, rhs))
    cons.append(LinearConstraint(rows2, -np.inf, -rhs))
    rows, hi = [], []
    for k in range(K):
        for f in range(F):
            r = np.zeros(nx + nd)
            for p in np.where(C[k] > 0)[0]:
                r[xi(p, f)] = 1
            rows.append(r)
            hi.append((C[k] > 0).sum() - 1)
    cons.append(LinearConstraint(np.array(rows), 0, hi))
    res = milp(c=c, constraints=cons, integrality=np.concatenate([np.ones(nx), np.zeros(nd)]),
               bounds=Bounds(np.zeros(nx + nd), np.concatenate([np.ones(nx), np.full(nd, np.inf)])),
               options={"time_limit": 900, "mip_rel_gap": 0.01})
    report(res, "dev folds")
    X = np.round(res.x[:nx]).reshape(P, F).astype(bool)
    pd.Series({projects[p]: int(np.argmax(X[p])) for p in range(P)}).rename("fold").to_csv(
        out / "optimal_devfolds_assignment.tsv", sep="\t")
    log.info("dev folds: %d training projects, %d balanced classes", P, K)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    hold = heldout(a.out)
    dev_folds(hold, a.out)


if __name__ == "__main__":
    main()
