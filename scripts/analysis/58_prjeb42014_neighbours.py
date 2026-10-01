#!/usr/bin/env python3
"""For each PRJEB42014 training run, the most similar training runs from OTHER studies
(cosine similarity of the unitig fraction vectors), summarised by the sample's collection
era. Also the same for the Brealey2020 bear runs as a reference. Writes
results/depth_vs_correctness_v9/prjeb42014_neighbours.tsv (per run, top 5) and prints the
era summary.
    sbatch on edid, 32 GB: ./env/bin/python scripts/analysis/58_prjeb42014_neighbours.py
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
K = 5
with np.load(ROOT / "data/matrices/matrix_v9_train/unitigs.frac.s8a.npz", allow_pickle=False) as z:
    ids = z["sample_ids"].astype(str); X = z["frac"].astype(np.float32)
meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[ids]
ena = pd.read_csv(ROOT / "results/depth_vs_correctness_v9/ena_PRJEB42014_runs.tsv", sep="\t").set_index("run_accession")
norm = np.linalg.norm(X, axis=1); norm[norm == 0] = 1; Xn = X / norm[:, None]
proj = meta.archive_project.to_numpy()
rows = []
for study in ("PRJEB42014", "PRJEB33363"):
    q = np.where(proj == study)[0]; others = np.where(proj != study)[0]
    S = Xn[q] @ Xn[others].T
    for qi, sims in zip(q, S):
        top = others[np.argsort(-sims)[:K]]
        for rank, j in enumerate(top):
            rows.append({"query_study": study, "run": ids[qi], "rank": rank + 1, "similarity": float(sims[np.where(others == j)[0][0]]),
                         "neighbour": ids[j], "neighbour_study": proj[j], "neighbour_project_name": meta.project_name.iloc[j],
                         "neighbour_host": meta.sample_host.iloc[j], "neighbour_material": meta.material.iloc[j], "neighbour_community": meta.community_type.iloc[j]})
nb = pd.DataFrame(rows)
nb.to_csv(ROOT / "results/depth_vs_correctness_v9/prjeb42014_neighbours.tsv", sep="\t", index=False)
nb["year"] = pd.to_numeric(ena.reindex(nb.run).collection_date.astype(str).str[:4].to_numpy(), errors="coerce")
nb["era"] = pd.cut(nb.year, [1800, 1900, 1950, 1990, 2020], labels=["1842-1900", "1901-1950", "1951-1990", "1991-2016"]).astype(str)
nb.loc[nb.query_study == "PRJEB33363", "era"] = "Brealey2020 bears (reference)"
nb["neighbour_kind"] = nb.neighbour_project_name.astype(str) + " | " + nb.neighbour_host.astype(str) + " | " + nb.neighbour_material.astype(str) + " | " + nb.neighbour_community.astype(str)
pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 70)
for era, g in nb.groupby("era"):
    print(f"\n=== {era}: {g.run.nunique()} runs, top-{K} neighbours ({len(g)} pairs), median similarity {g.similarity.median():.3f}")
    print(g.neighbour_kind.value_counts(normalize=True).head(6).round(3).to_string())
