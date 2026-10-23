#!/usr/bin/env python3
"""v19 C5 gate: re-projected training runs must reproduce their training-matrix columns.

`select` writes 20 training runs spread over the matrix columns: the first and the last
column, and 18 more at evenly spaced positions, each from a BioProject not yet used.
`check` compares each run's projected vector (13_project_v19.sbatch) with its column of
data/matrices/matrix_v17_train/unitigs.frac.mat; the gate passes only if every run's
maximum absolute difference is <= 1e-6. A failure stops the chain: no held-out run is
projected until the cause is found.

    ./env/bin/python scripts/validation/14_projection_gate_v19.py select --out data/project_v19_gate_runs.txt
    ./env/bin/python scripts/validation/14_projection_gate_v19.py check --runs data/project_v19_gate_runs.txt \\
        --vectors results/v19_projection_gate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MATRIX = ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat"
FOF = ROOT / "data/matrices/matrix_v17_train/kmtricks.fof"
META = ROOT / "data/splits_v19/train_metadata.tsv"
N_RUNS, TOL = 20, 1e-6

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def select(out: Path) -> None:
    fof = [l.split()[0] for l in open(FOF) if l.strip()]
    proj = pd.read_csv(META, sep="\t", dtype=str).set_index("Run_accession").archive_project
    chosen, used = [fof[0], fof[-1]], {proj[fof[0]], proj[fof[-1]]}
    for target in np.linspace(0, len(fof) - 1, N_RUNS)[1:-1]:
        order = sorted(range(len(fof)), key=lambda i: abs(i - target))
        i = next(i for i in order if proj[fof[i]] not in used and fof[i] not in chosen)
        chosen.append(fof[i])
        used.add(proj[fof[i]])
    if len(set(chosen)) != N_RUNS or len({proj[a] for a in chosen}) != N_RUNS:
        raise SystemExit("could not pick 20 runs from 20 distinct BioProjects")
    out.write_text("\n".join(chosen) + "\n")
    log.info("wrote %d runs from %d BioProjects to %s", len(chosen), len(used), out)


def check(runs: Path, vectors: Path) -> int:
    from diana.data.loader import MatrixLoader
    X, ids, _ = MatrixLoader(str(MATRIX)).load()
    col = {str(a): i for i, a in enumerate(ids)}
    ok, rows = True, []
    for acc in runs.read_text().split():
        v = np.loadtxt(vectors / acc / f"{acc}_unitig_fraction.txt")
        m = X[col[acc]].astype(float)
        if v.shape != m.shape:
            raise SystemExit(f"{acc}: {v.shape} projected vs {m.shape} in the matrix")
        diff = np.abs(v - m)
        rows.append({"run": acc, "max_abs_diff": float(diff.max()), "n_differ": int((diff > TOL).sum()),
                     "nonzero_projected": int((v > 0).sum()), "nonzero_matrix": int((m > 0).sum())})
        ok &= bool(diff.max() <= TOL)
    table = pd.DataFrame(rows)
    table.to_csv(vectors / "gate_report.tsv", sep="\t", index=False)
    print(table.to_string(index=False))
    log.info("projection gate %s (%d runs, tolerance %g)", "PASSED" if ok else "FAILED", len(rows), TOL)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--runs", type=Path, required=True)
    c.add_argument("--vectors", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.out)
        return 0
    return check(a.runs, a.vectors)


if __name__ == "__main__":
    raise SystemExit(main())
