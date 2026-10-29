#!/usr/bin/env python3
"""How much do the 23 unlabelled Klapper 2023 training runs (re-sequenced Fellows Yates 2021 samples, whose own runs are
held-out) shape the vocabulary? For each of the 85,986 unitigs, the number of training runs with a non-zero fraction, with and
without those runs, against the 10 % and 90 % prevalence bounds of the vocabulary filter (applied per unitig here as an
approximation of the k-mer-level filter). Writes results/klapper_check_v19/summary.tsv.

    ./env/bin/python scripts/analysis/80_klapper_vocabulary_v19.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.data.loader import MatrixLoader  # noqa: E402

lib = pd.read_csv(ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0/ancientmetagenome-hostassociated_libraries.tsv", sep="\t", dtype=str)
klapper = set(lib[lib.project_name == "Klapper2023"].archive_data_accession)
X, runs, _ = MatrixLoader(ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat").load()
runs = np.array([str(r) for r in runs])
k = np.isin(runs, list(klapper))
nz = (X > 0)
n_all, n_k = nz.sum(0), nz[k].sum(0)
N, Nk = len(runs), int(k.sum())
lo_all, hi_all = n_all / N, n_all / N
wo = (n_all - n_k) / (N - Nk)
rows = {"training runs": N, "Klapper 2023 runs": Nk,
        "unitigs with a non-zero fraction in any Klapper run": int((n_k > 0).sum()),
        "unitigs below 10 % without the Klapper runs": int((wo < 0.10).sum()),
        "unitigs above 90 % without the Klapper runs": int((wo > 0.90).sum()),
        "unitigs below 10 % with all runs (per-unitig approximation)": int((n_all / N < 0.10).sum()),
        "unitigs above 90 % with all runs (per-unitig approximation)": int((n_all / N > 0.90).sum())}
pd.Series(rows).to_csv(ROOT / "results/klapper_check_v19/summary.tsv", sep="\t", header=False)
print(pd.Series(rows).to_string())
