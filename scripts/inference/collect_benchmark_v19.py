#!/usr/bin/env python3
"""Runtime and memory of diana-predict with the released DIANA model (run_benchmark_v19.sbatch).

--select writes the run lists the sbatch reads (selection rule in each header): 30 held-out
runs for the Logan unitigs route (runs.txt) and 10 for the raw reads route (runs_reads.txt),
spread over input size.

Default: parses every results/benchmark_v19/<route>/<run>/time_v.txt (/usr/bin/time -v) into
results/benchmark_v19/benchmark.tsv: route, run, input_bytes, wall_clock_s, max_rss_kb, cpu_percent,
exit_status, max_abs_diff_paper (largest |difference| between the run's predicted probabilities
and results/heldout_v19/diana_refit/probs_DIANA_<task>.tsv, all tasks; NaN without predictions).
max_rss_kb is GNU time's peak resident set of the largest single process, not a sum over processes.
--net: the rerun with DIANA as the shared network alone (PROTOCOLS.md v19 amendment 2026-10-26), timed under
results/benchmark_v19net/ and compared with results/heldout_v19/diana_net/; run lists unchanged.

    ./env/bin/python scripts/inference/collect_benchmark_v19.py --select
    ./env/bin/python scripts/inference/collect_benchmark_v19.py [--net]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/benchmark_v19"
HELDOUT = ROOT / "results/v19_heldout/heldout_ids.txt"
LOGAN_DIR = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify/data/unitigs")
RAW_DIR = ROOT / "data/validation/raw"
PAPER = ROOT / "results/heldout_v19/diana_refit/probs_DIANA_{task}.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]
TIME_FIELDS = {"Elapsed (wall clock) time (h:mm:ss or m:ss)": "wall_clock_s",
               "Maximum resident set size (kbytes)": "max_rss_kb",
               "Percent of CPU this job got": "cpu_percent", "Exit status": "exit_status"}
RULE = ("# Selection rule (fixed before any timing): among the {n} held-out runs of\n"
        "# results/v19_heldout/heldout_ids.txt with {what}, sort by input size in bytes\n"
        "# (ties by accession) and take ranks round(q * (n - 1)) at q = i / ({k} - 1), i = 0..{k}-1.\n"
        "# Columns: run, input_bytes, input files (comma-separated). Array index i = data line i + 1.\n")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def reads_files(acc: str) -> list[Path]:
    """A single <acc>.fastq.gz or a complete <acc>_1/_2 pair; anything else is not used."""
    single, r1, r2 = (RAW_DIR / acc / f"{acc}{s}.fastq.gz" for s in ("", "_1", "_2"))
    if r1.exists() and r2.exists() and not single.exists():
        return [r1, r2]
    if single.exists() and not r1.exists() and not r2.exists():
        return [single]
    return []


def spread(sizes: pd.DataFrame, k: int) -> pd.DataFrame:
    s = sizes.sort_values(["input_bytes", "run"]).reset_index(drop=True)
    return s.iloc[sorted({int(round(q * (len(s) - 1))) for q in np.linspace(0, 1, k)})]


def write_list(path: Path, sizes: pd.DataFrame, k: int, what: str) -> None:
    pick = spread(sizes, k)
    with open(path, "w") as fh:
        fh.write(RULE.format(n=len(sizes), what=what, k=k))
        for r in pick.itertuples():
            fh.write(f"{r.run}\t{r.input_bytes}\t{r.files}\n")
    log.info("%s: %d runs, %d to %d bytes", path, len(pick), pick.input_bytes.min(), pick.input_bytes.max())


def select() -> None:
    accs = HELDOUT.read_text().split()
    logan = [(a, LOGAN_DIR / f"{a}.unitigs.fa.gz") for a in accs]
    missing = [a for a, f in logan if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} held-out runs without Logan unitigs, e.g. {missing[:3]}")
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "runs.txt").exists() or (OUT / "runs_reads.txt").exists():
        raise SystemExit(f"{OUT}: run lists exist; refusing to overwrite")
    write_list(OUT / "runs.txt", pd.DataFrame(
        [{"run": a, "input_bytes": f.stat().st_size, "files": str(f)} for a, f in logan]), 30,
        "Logan unitigs (.unitigs.fa.gz)")
    reads = [(a, reads_files(a)) for a in accs]
    write_list(OUT / "runs_reads.txt", pd.DataFrame(
        [{"run": a, "input_bytes": sum(f.stat().st_size for f in fs), "files": ",".join(map(str, fs))}
         for a, fs in reads if fs]), 10,
        "local raw reads in data/validation/raw (one <run>.fastq.gz or a complete _1/_2 pair)")


def parse_time(path: Path) -> dict:
    row = {}
    for line in path.read_text().splitlines():
        key, _, val = line.strip().rpartition(": ")
        if key in TIME_FIELDS:
            row[TIME_FIELDS[key]] = val.strip()
    if "wall_clock_s" in row:
        parts = [float(p) for p in row["wall_clock_s"].split(":")]
        row["wall_clock_s"] = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    if "cpu_percent" in row:
        row["cpu_percent"] = float(row["cpu_percent"].rstrip("%") or "nan")
    return row


def paper_diff(run: str, run_dir: Path, paper: dict) -> float:
    preds = list(run_dir.glob("*/*_predictions.json"))
    if len(preds) != 1:
        return float("nan")
    p = json.load(open(preds[0])).get("predictions", {})
    diffs = []
    for task in TASKS:
        if task not in p or run not in paper[task].index:
            return float("nan")
        ref = paper[task].loc[run]
        got = pd.Series(p[task]["probabilities"]).rename(lambda c: f"p_{c}")
        if set(got.index) != set(ref.index):
            return float("nan")
        diffs.append(float((got - ref[got.index]).abs().max()))
    return max(diffs)


def collect(runs_dir: Path, paper_path: Path) -> None:
    paper = {t: pd.read_csv(str(paper_path).format(task=t), sep="\t").set_index("Run_accession").filter(like="p_")
             for t in TASKS}
    rows = []
    for route, listing in (("logan", "runs.txt"), ("reads", "runs_reads.txt")):
        if not (OUT / listing).exists():
            continue
        runs = pd.read_csv(OUT / listing, sep="\t", comment="#", header=None, names=["run", "input_bytes", "files"])
        for r in runs.itertuples():
            t = runs_dir / route / r.run / "time_v.txt"
            row = {"route": route, "run": r.run, "input_bytes": r.input_bytes}
            row.update(parse_time(t) if t.exists() else {"exit_status": "not run"})
            row["max_abs_diff_paper"] = paper_diff(r.run, runs_dir / route / r.run, paper)
            rows.append(row)
    cols = ["route", "run", "input_bytes", "wall_clock_s", "max_rss_kb", "cpu_percent", "exit_status",
            "max_abs_diff_paper"]
    df = pd.DataFrame(rows).reindex(columns=cols)
    df.to_csv(runs_dir / "benchmark.tsv", sep="\t", index=False)
    log.info("%s: %d rows, %d finished with exit status 0", runs_dir / "benchmark.tsv", len(df),
             int((df.exit_status.astype(str) == "0").sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--select", action="store_true", help="write runs.txt and runs_reads.txt")
    ap.add_argument("--net", action="store_true", help="collect the network-only rerun (results/benchmark_v19net)")
    a = ap.parse_args()
    if a.select:
        select()
    elif a.net:
        collect(ROOT / "results/benchmark_v19net", ROOT / "results/heldout_v19/diana_net/probs_DIANA_{task}.tsv")
    else:
        collect(OUT, PAPER)


if __name__ == "__main__":
    main()
