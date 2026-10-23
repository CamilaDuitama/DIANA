"""Compute resources of building a sample's unitig vector with diana-predict's Logan route (v19 held-out set).

One SLURM array task per held-out run (job 26971202, scripts/validation/13_project_v19.sbatch: 8 CPUs, 32 GB
requested): wall time and peak memory from sacct (results/compute_v19/sacct_26971202.tsv), the run from the
task's log, the input size from its Logan unitig file. The network's prediction is not included. Writes
results/compute_v19/projection_resources.tsv, the supplementary figures results/paper_v19/sup_runtime.png and
sup_memory.png (plotly, DIANA's colour from config.py) and the manuscript table tables/sup_table_compute.tex.

    ./env/bin/python scripts/paper/generate_compute_resources_v19.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
JOB = "26971202"
RERUN = "26997044"   # the one run that exceeded 32 GB, rerun with 256 GB requested
LOGAN = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify/data/unitigs")
OUT = ROOT / "results/compute_v19"
FIG = ROOT / "results/paper_v19"
TABLES = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/tables")
COLOUR = PLOT_CONFIG["colors"]["models"]["DIANA"]


def sacct(job: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(OUT / f"sacct_{job}.tsv", sep="|", header=None, names=["id", "state", "seconds", "maxrss", "cpu", "node"], dtype=str)
    task = s[s.id.str.fullmatch(rf"{job}_\d+")].set_index("id")
    batch = s[s.id.str.endswith(".batch")].assign(id=lambda d: d.id.str.replace(".batch", "", regex=False)).set_index("id")
    return task, batch


def collect() -> pd.DataFrame:
    task, batch = sacct(JOB)
    rtask, rbatch = sacct(RERUN)
    rows = []
    for tid, r in task.iterrows():
        i = tid.split("_")[1]
        job = JOB
        if r.state == "OUT_OF_MEMORY" and f"{RERUN}_{i}" in rtask.index:
            job, tid, r, batch_ = RERUN, f"{RERUN}_{i}", rtask.loc[f"{RERUN}_{i}"], rbatch
        else:
            batch_ = batch
        log = ROOT / f"logs/project_v19/proj_{job}_{i}.out"
        m = re.match(r"=== (\S+) ===", open(log).readline()) if log.exists() else None
        if r.state != "COMPLETED" or m is None:
            raise ValueError(f"task {tid}: state {r.state}, log {'found' if m else 'missing'}")
        acc = m.group(1)
        rows.append({"Run_accession": acc, "task": int(i), "job": job, "minutes": int(r.seconds) / 60,
                     "peak_memory_gb": float(batch_.at[tid, "maxrss"].rstrip("M")) / 1024,
                     "input_mb": (LOGAN / f"{acc}.unitigs.fa.gz").stat().st_size / 1e6})
    d = pd.DataFrame(rows)
    if d.Run_accession.duplicated().any():
        raise ValueError("a run was projected twice")
    return d


def scatter(d: pd.DataFrame, y: str, ytitle: str, out: Path) -> None:
    fig = go.Figure(go.Scatter(x=d.input_mb, y=d[y], mode="markers", marker=dict(color=COLOUR, size=6, opacity=0.6), showlegend=False))
    fig.update_layout(template=PLOT_CONFIG["template"], width=560, height=440, font=dict(size=14, color="#333333"),
                      xaxis=dict(title="Logan unitig file (MB, compressed)", type="log"), yaxis=dict(title=ytitle),
                      margin=dict(l=70, r=20, t=30, b=60))
    fig.write_image(out, scale=2)


def table(d: pd.DataFrame) -> str:
    d = d.assign(q=pd.qcut(d.input_mb, 4, labels=["1st", "2nd", "3rd", "4th"]))
    lines = []
    for label, g in [*((f"{q} quarter", g) for q, g in d.groupby("q", observed=True)), ("all runs", d)]:
        lines.append(f"{label} & {len(g)} & {g.input_mb.min():.0f}--{g.input_mb.max():.0f} & {g.minutes.median():.1f} ({g.minutes.max():.1f}) & "
                     f"{g.peak_memory_gb.median():.1f} ({g.peak_memory_gb.max():.1f}) \\\\")
    lines.insert(-1, "\\midrule")
    return ("\\centering\n\\caption{Compute resources of building a run's unitig vector from its Logan unitigs with \\dianapredict, for the 863 "
            "held-out runs (8 CPUs and 32\\,GB requested per run; one run exceeded 32\\,GB and was rerun with more memory), by quarter of input size: runs, compressed input size, wall time and peak "
            "memory, median (maximum). The network's prediction is not included.}\n\\label{tab:resources}\n\\small\n\\begin{tabular}{lrrcc}\n\\toprule\n"
            "Input size & Runs & Input (MB) & Wall time (min) & Peak memory (GB) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    d = collect()
    d.to_csv(OUT / "projection_resources.tsv", sep="\t", index=False)
    FIG.mkdir(parents=True, exist_ok=True)
    scatter(d, "minutes", "wall time (min)", FIG / "sup_runtime.png")
    scatter(d, "peak_memory_gb", "peak memory (GB)", FIG / "sup_memory.png")
    (TABLES / "sup_table_compute.tex").write_text(table(d))
    print(f"{len(d)} runs; wall time median {d.minutes.median():.1f} min (max {d.minutes.max():.1f}); peak memory median "
          f"{d.peak_memory_gb.median():.1f} GB (max {d.peak_memory_gb.max():.1f}); input {d.input_mb.median():.0f} MB median; "
          f"Spearman time vs input {d[['minutes', 'input_mb']].corr('spearman').iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
