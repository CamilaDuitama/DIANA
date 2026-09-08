#!/usr/bin/env python
"""Can sequencing depth alone predict the metadata? (R3.7 / R3.8)

Question answered
-----------------
Referee 3 argues DIANA may be reading sequencing depth rather than biology: the
`--out-frac` feature is the fraction of a unitig's k-mers seen in a sample, which
rises monotonically with depth, and k=31 with min-abundance 2 penalises shallow
libraries. If a model given *only* depth matches one given 78,430 unitig
features, the features are not doing the work.

This is the negative control that answer needs. It is deliberately the cheapest
possible experiment: AncientMetagenomeDir records `read_count` for every run
(100 % coverage on v9), so no sequencing data has to be touched.

Read it as a floor, not a competitor. Depth scoring near the majority-class rate
means the unitig features carry real signal; depth approaching the full model
means they do not.

Inputs : data/splits_v9/, AncientMetagenomeDir libraries tables
Outputs: results/depth_only_control/{metrics.json,summary.txt}
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AMD = PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"
SPLITS = PROJECT_ROOT / "data/splits_v9"
TARGETS = ["community_type", "feature", "sample_host", "material"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def depth_table() -> pd.DataFrame:
    lib = pd.concat([pd.read_csv(AMD / f"ancientmetagenome-{k}_libraries.tsv",
                                 sep="\t", low_memory=False)
                     for k in ("hostassociated", "environmental")], ignore_index=True)
    lib = lib.drop_duplicates("archive_data_accession")
    out = pd.DataFrame({
        "Run_accession": lib.archive_data_accession,
        "read_count": pd.to_numeric(lib.read_count, errors="coerce"),
        "download_size": pd.to_numeric(
            lib.download_sizes.astype(str).str.split(";").str[0], errors="coerce"),
    })
    # Depth spans three orders of magnitude, so model it on a log scale; a linear
    # model on raw counts would be driven entirely by the few huge libraries.
    out["log_reads"] = np.log10(out.read_count.clip(lower=1))
    out["log_size"] = np.log10(out.download_size.clip(lower=1))
    return out.set_index("Run_accession")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/depth_only_control")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    depth = depth_table()
    tr = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t")
    te = pd.read_csv(SPLITS / "test_metadata.tsv", sep="\t")
    el = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")

    feats = ["log_reads", "log_size"]
    results, lines = {}, ["Depth-only control (R3.7 / R3.8)", "",
                          "Features: log10(read_count), log10(download_size). Nothing else.", ""]

    for target in TARGETS:
        eligible = set(el[(el.target == target) & el.evaluable]["class"])
        a = tr[tr[target].notna()].join(depth, on="Run_accession")
        b = te[te[target].notna()].join(depth, on="Run_accession")
        a, b = a.dropna(subset=feats), b.dropna(subset=feats)
        seen = set(a[target])
        b = b[b[target].isin(seen)]            # out-of-vocabulary excluded, as elsewhere
        if len(b) < 10:
            continue
        Xtr, ytr = a[feats].to_numpy(), a[target].astype(str).to_numpy()
        Xte, yte = b[feats].to_numpy(), b[target].astype(str).to_numpy()
        elig = sorted(c for c in set(yte) if c in eligible)

        models = {
            "MajorityClass": DummyClassifier(strategy="most_frequent"),
            "Depth_LogReg": make_pipeline(StandardScaler(),
                                          LogisticRegression(max_iter=2000,
                                                             class_weight="balanced")),
            "Depth_RandomForest": RandomForestClassifier(n_estimators=300,
                                                         random_state=args.seed, n_jobs=-1),
        }
        results[target] = {"n_train": len(ytr), "n_test": len(yte),
                           "n_eligible_classes": len(elig)}
        lines.append(f"{target}  (train {len(ytr)}, test {len(yte)}, {len(elig)} eligible classes)")
        lines.append(f"    {'model':22s} {'acc':>7s} {'bal_acc':>8s} {'F1_elig':>8s}")
        for name, m in models.items():
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            r = {"accuracy": float(accuracy_score(yte, p)),
                 "balanced_accuracy": float(balanced_accuracy_score(yte, p)),
                 "f1_macro_eligible": float(f1_score(yte, p, labels=elig, average="macro",
                                                     zero_division=0)) if elig else float("nan")}
            results[target][name] = r
            lines.append(f"    {name:22s} {r['accuracy']:7.3f} {r['balanced_accuracy']:8.3f} "
                         f"{r['f1_macro_eligible']:8.3f}")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    json.dump(results, open(args.output / "metrics.json", "w"), indent=2)
    (args.output / "summary.txt").write_text(report + "\n")
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
