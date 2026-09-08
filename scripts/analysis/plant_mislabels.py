#!/usr/bin/env python
"""Plant realistic mislabels, so anomaly detection can be scored as an ROC (step 21).

Question answered
-----------------
The real curator-corrected errors we can use number nine (Rigou2022, step 20) --
enough for a story, far too few for a detection rate. So corrupt labels
deliberately, at a known rate, and measure how many the detector catches against
how often it fires on clean samples.

Why the corruption model matters more than the rate
---------------------------------------------------
Uniform random relabelling flatters the result. Turning an `oral` sample into
`thermokarst` is trivially catchable and would report a detection rate that says
nothing about real use. Real curator errors are confusions between *adjacent*
categories, and we have a measured example of exactly that: the merge-key join
bug produced 154 real mislabels whose direction is known.

    community_type   oral -> skeletal tissue (59), skeletal tissue -> oral (10),
                     skeletal tissue -> soft tissue (10), ...
    material         dental calculus -> tooth (48), tooth -> dental calculus (9),
                     dental calculus -> bone (8), ...

`--model empirical` samples from that distribution. It only covers the classes
the bug happened to touch, so `--model plausible` is the fallback: swap within
the same branch (host-associated or environmental), which keeps the error
biologically conceivable without pretending to be measured. `--model uniform` is
provided only as the optimistic bound to report alongside, never on its own.

Inputs : data/splits_v9/*_metadata.tsv, results/amd_label_corrections/mergekey_usable.tsv
Outputs: <output>/planted_<split>.tsv  — metadata with corrupted labels plus
         `<target>_planted` flags marking ground truth, and planting_report.txt
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = PROJECT_ROOT / "data/splits_v9"
MERGEKEY = PROJECT_ROOT / "results/amd_label_corrections/mergekey_usable.tsv"
TARGETS = ["community_type", "feature", "sample_host", "material"]
HOST_TARGETS = {"community_type", "sample_host"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def empirical_transitions() -> dict:
    """Observed error directions, from the merge-key corrections.

    Keyed target -> true class -> (candidate wrong classes, probabilities).
    Note the direction: `old` is what the wrong metadata said, `new` is the truth,
    so a realistic corruption of `new` is `old`.
    """
    if not MERGEKEY.exists():
        logger.warning("%s absent; empirical model unavailable", MERGEKEY)
        return {}
    df = pd.read_csv(MERGEKEY, sep="\t")
    out: dict = {}
    for (field, truth), g in df.groupby(["field", "new"]):
        counts = g["old"].value_counts()
        out.setdefault(field, {})[truth] = (counts.index.to_numpy(),
                                            (counts / counts.sum()).to_numpy())
    return out


def confusable_classes(train: pd.DataFrame, target: str, eligible: set,
                       top_k: int = 3) -> dict:
    """For each class, the classes it could plausibly be confused with.

    Drawing uniformly from the whole target is what makes a planted-mislabel
    experiment dishonest: relabelling a `Homo sapiens` sample as
    `Ambrosia artemisiifolia` (ragweed) is not a mistake anyone makes, and any
    detector catches it, so the reported detection rate is inflated.

    Confusability is estimated from context: two classes are confusable if they
    occur alongside similar values of the *other* targets. `tooth` and
    `dental calculus` share oral, human, skeletal contexts; ragweed shares none of
    them with humans. Cosine similarity over that co-occurrence profile, nearest
    top_k retained. Falls back to the full pool for classes with no context.
    """
    others = [t for t in TARGETS if t != target and t in train]
    sub = train[train[target].notna()]
    classes = sorted(c for c in sub[target].unique() if c in eligible)
    if len(classes) < 2:
        return {c: np.array([]) for c in classes}

    # Profile each class by how often it co-occurs with every value of every other target.
    cols = []
    for o in others:
        cols += [f"{o}={v}" for v in sorted(sub[o].dropna().unique())]
    profile = pd.DataFrame(0.0, index=classes, columns=cols or ["_none"])
    for c in classes:
        rows = sub[sub[target] == c]
        for o in others:
            for v, n in rows[o].dropna().value_counts().items():
                profile.at[c, f"{o}={v}"] = n
    norm = np.linalg.norm(profile.to_numpy(), axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    sim = (profile.to_numpy() / norm) @ (profile.to_numpy() / norm).T
    np.fill_diagonal(sim, -np.inf)

    out = {}
    for i, c in enumerate(classes):
        order = np.argsort(sim[i])[::-1]
        near = [classes[j] for j in order[:top_k] if np.isfinite(sim[i][j]) and sim[i][j] > 0]
        out[c] = np.array(near if near else [x for x in classes if x != c])
    return out


def plant(df: pd.DataFrame, train: pd.DataFrame, eligible: dict, rate: float,
          model: str, rng: np.random.Generator, report: list) -> pd.DataFrame:
    out = df.copy()
    emp = empirical_transitions() if model in ("empirical", "mixed") else {}

    for target in TARGETS:
        if target not in out:
            continue
        flag = f"{target}_planted"
        out[flag] = False
        # Only corrupt rows that have a label and whose class is evaluable -- a
        # planted error on a class the model cannot emit is not a detectable error.
        ok = out[target].notna() & out[target].isin(eligible.get(target, set()))
        idx = out.index[ok]
        if len(idx) == 0:
            report.append(f"  {target:16s} no eligible rows")
            continue
        n = int(round(rate * len(idx)))
        chosen = rng.choice(idx, size=n, replace=False) if n else np.array([], dtype=int)

        confusable = confusable_classes(train, target, eligible.get(target, set()))
        n_emp = 0
        for i in chosen:
            truth = out.at[i, target]
            cand, prob = emp.get(target, {}).get(truth, (None, None))
            if cand is not None and len(cand):
                new = rng.choice(cand, p=prob)
                n_emp += 1
            else:
                alt = confusable.get(truth, np.array([]))
                alt = alt[alt != truth]
                if not len(alt):
                    continue
                new = rng.choice(alt)
            out.at[i, target] = new
            out.at[i, flag] = True
        planted = int(out[flag].sum())
        report.append(f"  {target:16s} {planted:5d} of {len(idx):5d} eligible rows corrupted "
                      f"({100 * planted / len(idx):.1f}%), {n_emp} from the empirical model")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--rate", type=float, default=0.10, help="fraction of eligible rows to corrupt")
    ap.add_argument("--model", default="mixed", choices=["empirical", "plausible", "mixed", "uniform"],
                    help="mixed = empirical where measured, plausible elsewhere")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/planted_mislabels")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLITS / f"{args.split}_metadata.tsv", sep="\t")
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t")
    el = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    eligible = {t: set(g[g.evaluable]["class"]) for t, g in el.groupby("target")}
    if args.model == "uniform":
        # No restriction to evaluable classes is the point of the uniform bound,
        # but keep it inside the label space or the model cannot express the error.
        eligible = {t: set(train[t].dropna().unique()) for t in TARGETS if t in train}

    rng = np.random.default_rng(args.seed)
    report = [f"Planted mislabels — split={args.split}, rate={args.rate}, "
              f"model={args.model}, seed={args.seed}", ""]
    out = plant(df, train, eligible, args.rate, args.model, rng, report)

    dest = args.output / f"planted_{args.split}_{args.model}_r{args.rate}.tsv"
    out.to_csv(dest, sep="\t", index=False)
    report.append("")
    report.append(f"wrote {dest.name}")
    report.append("Ground truth is the `<target>_planted` boolean columns.")
    text = "\n".join(report)
    print(text)
    (args.output / "planting_report.txt").write_text(text + "\n")
    json.dump({"split": args.split, "rate": args.rate, "model": args.model,
               "seed": args.seed, "file": dest.name},
              open(args.output / "planting_config.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
