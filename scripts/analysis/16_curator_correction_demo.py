#!/usr/bin/env python3
"""Would DIANA have caught the errors a curator caught by hand? (R3.6)

The planted-mislabel experiment gives a rate but the labels are synthetic. This is
the complement: real errors, found by AncientMetagenomeDir curators and fixed in the
repository's git history, on runs that are in **held-out** so the model never saw
either version.

Each run is presented with its **pre-correction** label, the one a curator would
have been looking at. Two things are then read off the model's probabilities:

    P(old)  what the model gives the label that was on file
    P(new)  what it gives the corrected label

A run counts as caught if the model's top prediction is the corrected label. It
counts as flagged at a threshold if 1 - P(old) exceeds it, which is the same
detector `15_anomaly_detection_roc.py` scores.

What this can and cannot claim
------------------------------
These are a handful of runs from one or two correction events, so this is a
**demonstration, never a rate**. The rate comes from planted mislabels. Reporting
this as a detection rate would be reporting n=20 from two studies.

`feature` is the stronger case: the wrong label (`lake`) is common in training and
the right one (`thermokarst`) is rare, so choosing the rare class means going
against the prior. A correction toward a *more* common class is weaker evidence,
because the prior alone pushes that way.

    ./env/bin/python scripts/analysis/16_curator_correction_demo.py \\
        --predictions results/final_eval_v9/test_predictions.tsv --label DIANA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
CORRECTIONS = PROJECT_ROOT / "results/amd_label_corrections/corrections.tsv"
TASKS = ["community_type", "feature", "sample_host", "material"]


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    if df.shape[1] == 1:
        raise SystemExit(f"{path} parsed to one column with sep={sep!r}")
    return df


def class_index(pred: pd.DataFrame, task: str) -> dict:
    idx_col = f"{task}_true_idx"
    if idx_col not in pred.columns:
        raise SystemExit(f"{idx_col} missing; cannot map class names to probabilities")
    pairs = pred[[idx_col, f"{task}_true"]].dropna().drop_duplicates()
    return {r[1]: int(r[0]) for r in pairs.itertuples(index=False)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--label", default="DIANA")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="flag if 1 - P(old label) exceeds this")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/curator_demo")
    args = ap.parse_args()

    pred = read_table(args.predictions)
    corr = pd.read_csv(CORRECTIONS, sep="\t")
    held = set(Path(SPLITS / "test_accessions.txt").read_text().split())
    train = set(Path(SPLITS / "train_accessions.txt").read_text().split())

    # Only genuine relabels count. Vocabulary normalisation ("calculus" ->
    # "dental calculus") is a spelling fix, not an error a model could catch.
    corr = corr[(corr.change_type == "genuine relabel") & corr.archive_data_accession.notna()]
    corr = corr[corr.field.isin(TASKS)]
    in_train = corr[corr.archive_data_accession.isin(train)].archive_data_accession.nunique()
    corr = corr[corr.archive_data_accession.isin(held)]
    if corr.empty:
        raise SystemExit("no genuine relabels land in held-out")
    print(f"{corr.archive_data_accession.nunique()} held-out runs with genuine relabels "
          f"({in_train} more are in train and cannot be used)")

    n_train_lab = {t: pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t",
                                  low_memory=False)[t].value_counts() for t in TASKS}

    rows = []
    for task, g in corr.groupby("field"):
        if f"{task}_true" not in pred.columns:
            print(f"  {task}: not in the predictions file, skipping")
            continue
        idx_by_name = class_index(pred, task)
        prob_cols = sorted((c for c in pred.columns if c.startswith(f"{task}_prob_")),
                           key=lambda c: int(c.rsplit("_", 1)[1]))
        if not prob_cols:
            raise SystemExit(f"no {task}_prob_* columns; re-run diana-test")
        P = pred.set_index("Run_accession")[prob_cols]
        top = pred.set_index("Run_accession")[f"{task}_pred"]
        for r in g.itertuples(index=False):
            acc = r.archive_data_accession
            if acc not in P.index:
                continue
            p = P.loc[acc].to_numpy()
            i_old, i_new = idx_by_name.get(r.old), idx_by_name.get(r.new)
            p_old = float(p[i_old]) if i_old is not None and i_old < len(p) else np.nan
            p_new = float(p[i_new]) if i_new is not None and i_new < len(p) else np.nan
            rows.append({
                "task": task, "run": acc, "old_label": r.old, "new_label": r.new,
                "n_train_old": int(n_train_lab[task].get(r.old, 0)),
                "n_train_new": int(n_train_lab[task].get(r.new, 0)),
                "diana_pred": top.loc[acc], "p_old": p_old, "p_new": p_new,
                "caught": bool(top.loc[acc] == r.new),
                "flagged": bool(not np.isnan(p_old) and (1 - p_old) > args.threshold),
                "against_prior": bool(n_train_lab[task].get(r.new, 0)
                                      < n_train_lab[task].get(r.old, 0)),
            })

    if not rows:
        raise SystemExit("no corrected held-out run appears in the predictions file")

    df = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / f"curator_demo_{args.label}.tsv", sep="\t", index=False)

    lines = [f"Curator-corrected runs — {args.label}  (R3.6)", "",
             "Each run is shown with its PRE-correction label. P(old) is what the",
             "model gives the label that was on file; P(new) the corrected one.",
             "'caught' = the model's top prediction is the corrected label.",
             f"'flagged' = 1 - P(old) > {args.threshold}.", ""]
    for task, g in df.groupby("task"):
        lines.append(f"{task}  ({g.run.nunique()} runs)")
        lines.append(f"    {'old -> new':<34}{'n_train':>14}{'caught':>9}{'flagged':>9}"
                     f"{'P(old)':>9}{'P(new)':>9}")
        for (o, n), gg in g.groupby(["old_label", "new_label"]):
            nt = f"{gg.n_train_old.iloc[0]}->{gg.n_train_new.iloc[0]}"
            p_old = "n/a" if gg.p_old.isna().all() else f"{gg.p_old.mean():.3f}"
            lines.append(f"    {o + ' -> ' + n:<34}{nt:>14}"
                         f"{f'{gg.caught.sum()}/{len(gg)}':>9}"
                         f"{f'{gg.flagged.sum()}/{len(gg)}':>9}"
                         f"{p_old:>9}{gg.p_new.mean():>9.3f}")
        lines.append("")
    oov = df[df.p_old.isna()]
    ap_ = df[df.against_prior]
    lines += [f"Overall: {df.caught.sum()}/{len(df)} corrections caught, "
              f"{df.flagged.sum()}/{len(df)} flagged.",
              f"Of those going AGAINST the training prior (corrected class rarer than "
              f"the wrong one): {ap_.caught.sum()}/{len(ap_)} caught.",
              "",
              f"P(old) is n/a for {len(oov)} of {len(df)} corrections: the pre-correction",
              "label is not in the model's vocabulary at all, so no probability can be",
              "assigned to it and the flag score is undefined. Those runs cannot be",
              "scored by this detector, and their 0/n 'flagged' is an absence of a",
              "score, not model confidence. Only the corrections whose old label the",
              "model knows are scorable.",
              "",
              "This is a DEMONSTRATION, not a rate: a handful of runs from one or two",
              "correction events. The rate comes from planted mislabels",
              "(15_anomaly_detection_roc.py). Corrections toward a rarer class are the",
              "stronger evidence, since the prior pushes the other way."]
    report = "\n".join(lines)
    print("\n" + report)
    (args.output / f"summary_{args.label}.txt").write_text(report + "\n")
    json.dump(rows, open(args.output / f"curator_demo_{args.label}.json", "w"), indent=2)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
