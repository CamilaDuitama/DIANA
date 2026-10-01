#!/usr/bin/env python3
"""Phase 4, understanding only (never for choosing): per-study accuracy on the watch list and
accuracy on shallow evaluation runs, starting model against a candidate, from pooled out-of-fold
predictions in the 54_seed_ensemble.py layout (oof_<task>_<task>.tsv).

Watch list (4.2): PRJNA354503 (tooth), PRJEB42014 (bears), marine-sediment studies PRJEB46821,
PRJNA1211513, PRJNA766251, PRJNA861836. Shallow (4.3): fewer than 600 non-zero unitigs
(results/depth_vs_correctness_v9/nonzero_counts.tsv), per class, for tooth, bone, skeletal tissue.

    ./env/bin/python scripts/analysis/63_watchlist_shallow_report.py --a results/reference_v9_oof --b results/step4_2_studyweight_v9_oof --name-b step4_2 --out results/step4_2_studyweight_v9
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
WATCH = {"PRJNA354503": "tooth study", "PRJEB42014": "bears", "PRJEB46821": "marine sediment", "PRJNA1211513": "marine sediment",
         "PRJNA766251": "marine sediment", "PRJNA861836": "marine sediment"}
SHALLOW, SHALLOW_CLASSES = 600, ["tooth", "bone", "skeletal tissue"]


def load(base: Path, task: str) -> pd.DataFrame:
    d = pd.read_csv(base / f"oof_{task}_{task}.tsv", sep="\t").set_index("Run_accession")
    d = d[d[f"{task}_true"].notna()]
    return pd.DataFrame({"true": d[f"{task}_true"].astype(str), "pred": d[f"{task}_pred"].astype(str)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="starting model OOF directory")
    ap.add_argument("--b", type=Path, required=True, help="candidate OOF directory")
    ap.add_argument("--name-a", default="reference"); ap.add_argument("--name-b", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    depth = pd.read_csv(ROOT / "results/depth_vs_correctness_v9/nonzero_counts.tsv", sep="\t").set_index("Run_accession")["n_nonzero"]
    watch_rows, shallow_rows = [], []
    for task in TASKS:
        a, b = load(args.a, task), load(args.b, task)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        proj = meta.loc[common, "archive_project"].astype(str)
        for p, what in WATCH.items():
            m = (proj == p).to_numpy()
            if m.sum():
                watch_rows.append({"task": task, "study": p, "what": what, "n_runs": int(m.sum()),
                                   f"acc_{args.name_a}": float((a.true[m] == a.pred[m]).mean()), f"acc_{args.name_b}": float((b.true[m] == b.pred[m]).mean())})
        shallow = (depth.reindex(common).fillna(0) < SHALLOW).to_numpy()
        for c in SHALLOW_CLASSES:
            m = shallow & (a.true == c).to_numpy()
            if m.sum():
                shallow_rows.append({"task": task, "class": c, "n_shallow_runs": int(m.sum()),
                                     f"acc_{args.name_a}": float((a.pred[m] == c).mean()), f"acc_{args.name_b}": float((b.pred[m] == c).mean())})
        m_all = shallow
        shallow_rows.append({"task": task, "class": "(all shallow runs)", "n_shallow_runs": int(m_all.sum()),
                             f"acc_{args.name_a}": float((a.true[m_all] == a.pred[m_all]).mean()) if m_all.sum() else float("nan"),
                             f"acc_{args.name_b}": float((b.true[m_all] == b.pred[m_all]).mean()) if m_all.sum() else float("nan")})
    w, s = pd.DataFrame(watch_rows), pd.DataFrame(shallow_rows)
    w.to_csv(args.out / f"watchlist_{args.name_b}.tsv", sep="\t", index=False); s.to_csv(args.out / f"shallow_{args.name_b}.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("watch list, per-study accuracy (understanding only):"); print(w.round(3).to_string(index=False))
    print(f"\nshallow evaluation runs (< {SHALLOW} non-zero unitigs), accuracy per class (understanding only):"); print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
