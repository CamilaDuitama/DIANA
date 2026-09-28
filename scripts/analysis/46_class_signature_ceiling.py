#!/usr/bin/env python3
"""S8.6, Experiment A: how many exact k-mers could ever identify each class across studies?

On the 2,716 training sketches at scaled=100 (1 k-mer in 100), vocabulary removed, for
every eligible class with at least two training BioProjects, count the off-list k-mers
that pass three tests at once:
  consistent   present in >= FRAC_IN_CLASS of the class's training runs;
  cross-study  present in >= 1 run of each of >= 2 of the class's training projects
               (loose), and present in >= FRAC_IN_CLASS of the runs of each of >= 2 of
               its projects (strict);
  exclusive    present in < FRAC_OUTSIDE of the labelled runs of other classes.
Single-project classes are reported without the cross-study test, i.e. what they would
have if a second study existed. Nothing here uses held-out or fits a model.

Output: results/s8a_probe_v9/class_signature_ceiling_k{K}.tsv, and the k-mer lists per
class (hash arrays, npz) for S8.8.

    ./env/bin/python scripts/analysis/46_class_signature_ceiling.py --ksize 31
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SK = ROOT / "data/sketches_v9"
SPLITS = ROOT / "data/splits_v9"
OUT = ROOT / "results/s8a_probe_v9"
TASKS = ["community_type", "feature", "sample_host", "material"]
FRAC_IN_CLASS, FRAC_OUTSIDE, MIN_RUNS = 1 / 3, 0.05, 2


def read_sketches(zip_path: Path, ksize: int) -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n = info.filename
            if not (n.endswith(".sig") or n.endswith(".sig.gz")):
                continue
            raw = z.read(info)
            if n.endswith(".gz"):
                raw = gzip.decompress(raw)
            for rec in json.load(io.BytesIO(raw)):
                for s in rec.get("signatures", []):
                    if s.get("ksize") == ksize:
                        out[rec["name"]] = np.array(s["mins"], dtype=np.uint64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ksize", type=int, default=31)
    a = ap.parse_args()
    t0 = time.time()
    V = np.sort(next(iter(read_sketches(SK / "vocab_v9_s100.sig.zip", a.ksize).values())))
    sk = read_sketches(SK / "unitigs_train_v9_s100.zip", a.ksize)
    train = [l.strip() for l in open(SPLITS / "train_accessions.txt") if l.strip()]
    missing = [x for x in train if x not in sk]
    if missing:
        raise SystemExit(f"{len(missing)} training runs without a sketch, e.g. {missing[:3]}")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession").loc[train]
    proj = meta["archive_project"].astype(str).to_numpy()
    logger.info("k=%d: %d sketches, vocabulary %d hashes, read in %.0f s", a.ksize, len(sk), len(V), time.time() - t0)
    off = {x: sk[x][~np.isin(sk[x], V, assume_unique=True)] for x in train}
    logger.info("off-list per run: median %d", int(np.median([len(v) for v in off.values()])))

    elig = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    elig = elig[elig.evaluable]
    rows, lists = [], {}
    for task in TASKS:
        lab = meta[task].astype(object)
        labelled = lab.notna().to_numpy()
        for cls in sorted(set(elig[elig.target == task]["class"].astype(str))):
            in_cls = (lab.astype(str).to_numpy() == cls) & labelled
            runs = np.where(in_cls)[0]
            if len(runs) < MIN_RUNS:
                rows.append({"task": task, "class": cls, "train_runs": int(len(runs)), "train_projects": 0})
                continue
            # candidates: consistent within the class
            H = np.concatenate([off[train[i]] for i in runs])
            u, c = np.unique(H, return_counts=True)
            cand = u[c >= max(1, FRAC_IN_CLASS * len(runs))]
            # per-project presence among the class's runs
            cls_projects = sorted(set(proj[runs]))
            per_proj_any, per_proj_frac = [], []
            for p in cls_projects:
                pr = [i for i in runs if proj[i] == p]
                Hp = np.concatenate([off[train[i]] for i in pr])
                up, cp = np.unique(Hp, return_counts=True)
                hit = np.isin(cand, up, assume_unique=True)
                per_proj_any.append(hit)
                frac = np.zeros(len(cand)); pos = np.searchsorted(up, cand[hit]); frac[hit] = cp[pos] / len(pr)
                per_proj_frac.append(frac >= FRAC_IN_CLASS)
            n_any = np.sum(per_proj_any, axis=0) if per_proj_any else np.zeros(len(cand))
            n_str = np.sum(per_proj_frac, axis=0) if per_proj_frac else np.zeros(len(cand))
            # exclusivity: presence among OTHER labelled runs of this task
            others = np.where(labelled & ~in_cls)[0]
            cnt_out = np.zeros(len(cand), dtype=np.int64)
            for i in others:
                hit = np.isin(cand, off[train[i]], assume_unique=True)
                cnt_out[hit] += 1
            excl = cnt_out < FRAC_OUTSIDE * max(1, len(others))
            multi = len(cls_projects) >= 2
            keep_loose = excl & (n_any >= 2) if multi else excl
            keep_strict = excl & (n_str >= 2) if multi else excl
            rows.append({"task": task, "class": cls, "train_runs": int(len(runs)), "train_projects": len(cls_projects),
                         "consistent": int(len(cand)), "consistent_and_exclusive": int(excl.sum()),
                         "signature_loose": int(keep_loose.sum()), "signature_strict": int(keep_strict.sum()),
                         "cross_study_tested": multi})
            lists[f"{task}|{cls}"] = cand[keep_loose]
            logger.info("%s / %s: runs %d, projects %d, consistent %d, exclusive %d, loose %d, strict %d",
                        task, cls, len(runs), len(cls_projects), len(cand), int(excl.sum()),
                        int(keep_loose.sum()), int(keep_strict.sum()))
    df = pd.DataFrame(rows).sort_values(["task", "train_runs"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"class_signature_ceiling_k{a.ksize}.tsv", sep="\t", index=False)
    np.savez_compressed(OUT / f"class_signatures_k{a.ksize}.npz", **{k.replace("|", "__").replace(" ", "_"): v for k, v in lists.items()})
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    logger.info("done in %.0f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
