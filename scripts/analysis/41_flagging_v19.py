#!/usr/bin/env python3
"""v19 mislabel-detection scores, dev folds or held-out (PROTOCOLS.md v19, C2 and C6).

Pool. For each task and planting, a run enters the flagging pool only if it had a label,
its TRUE class has training rows in the model's fit (dev fold: the other four folds;
held-out: all training runs; C2a), and its STATED label (planted where planted) is such
a class too. Other labelled runs are counted (`not_assessable_mean`), not scored. The
pool is built from the split alone, so every model is scored on the same runs.

Scores, per model, averaged over the plantings: recall at 5 / 10 / 20 % false flags
(a planting where the budget is not attainable is "can't check" and left out of the mean;
NaN when no planting reaches it), the choosing score (mean recall over the model's
attainable budgets; plantings where no budget is attainable are counted in
`plantings_cant_check`), ROC-AUC (threshold-free,
reported, never used for choosing), the share of correctly labelled runs tied at the
top, and the tooth-excluded recall at 5 % (explains, never decides). Intervals: 2,000
BioProject resamples shared by every model, planting and pair; a model's attainable
budgets are fixed per planting on the full data and reused on every resample; `*_draws`
columns give how many resamples produced a value. Paired comparisons a - b
use, per planting, the budgets attainable by both models on the full data, reused on
every resample. Definitions in diana.evaluation.scoring.

Checks: the planting file holds exactly the split's runs and was made from --split-dir;
non-planted stated labels equal the metadata and planted ones differ; a probability file
with a `fold` column agrees with dev_folds.tsv; every pooled run has probabilities.

    ./env/bin/python scripts/analysis/41_flagging_v19.py --split-dir data/splits_v19 \\
        --plantings results/planted_v19/dev --pattern 'planted_train_mixed_r0.1_s*.tsv' \\
        --model DIANA=results/.../probs_DIANA_{task}.tsv --model LogReg=... --pair DIANA:LogReg \\
        --out results/flagging_v19/dev_single
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
log = logging.getLogger("flagging_v19")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_probs(path_tpl: str, task: str, folds: pd.Series | None) -> pd.DataFrame:
    path = path_tpl.format(task=task)
    d = pd.read_csv(path, sep="\t").set_index("Run_accession")
    if folds is not None and "fold" in d.columns:
        bad = (d["fold"] != folds.reindex(d.index)).sum()
        if bad:
            raise ValueError(f"{path}: {bad} runs carry a fold other than dev_folds.tsv")
    cols = [c for c in d.columns if c.startswith("p_")]
    if not cols:
        raise ValueError(f"{path}: no probability columns")
    return d[cols].rename(columns=lambda c: c[2:])


def read_plantings(files: list[str], split_dir: Path, runs: pd.Index,
                   legacy_split: Path | None = None) -> list[pd.DataFrame]:
    """`legacy_split`: plantings made before provenance was recorded (v9), accepted only
    when the caller names the split they were made on and it equals split_dir."""
    out = []
    for f in files:
        if legacy_split is not None:
            if legacy_split.resolve() != split_dir.resolve():
                raise ValueError(f"legacy plantings declared for {legacy_split}, scored on {split_dir}")
        else:
            cfg_path = Path(f).parent / f"planting_config_{Path(f).stem}.json"
            if not cfg_path.exists():
                raise FileNotFoundError(f"{cfg_path}: planting provenance missing")
            made_from = json.load(open(cfg_path)).get("splits_dir")
            if made_from is None or Path(made_from).resolve() != split_dir.resolve():
                raise ValueError(f"{f} was planted on {made_from}, not on {split_dir}")
        pl = pd.read_csv(f, sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
        if set(pl.index) != set(runs):
            raise ValueError(f"{f}: {len(set(runs) - set(pl.index))} split runs missing, "
                             f"{len(set(pl.index) - set(runs))} extra")
        for t in TASKS:
            if f"{t}_planted" in pl:
                pl[f"{t}_planted"] = pl[f"{t}_planted"].map({"True": True, "False": False})
        out.append(pl.loc[runs])
    return out


def pool(meta: pd.DataFrame, pl: pd.DataFrame, task: str, assessable) -> tuple[pd.DataFrame, int]:
    """Pool rows (stated label, planted flag, project, material) and the count left out."""
    flag = f"{task}_planted"
    true, stated, planted = meta[task], pl[task], pl[flag]
    had = true.notna()
    ok = had & stated.notna() & planted.notna()
    ok &= pd.Series([assessable(r, str(t)) and assessable(r, str(s))
                     for r, t, s in zip(meta.index, true, stated)], index=meta.index)
    sub = pd.DataFrame({"true": true[ok].astype(str), "stated": stated[ok].astype(str),
                        "planted": planted[ok].astype(bool), "project": meta.loc[ok, "archive_project"].astype(str),
                        "material": meta.loc[ok, "material"].astype(str)})
    if ((sub.stated != sub.true) != sub.planted).any():
        raise ValueError(f"{task}: a stated label disagrees with the planted flag")
    return sub, int(had.sum() - ok.sum())


def flag_scores(table: pd.DataFrame, P: pd.DataFrame) -> np.ndarray:
    missing = table.index.difference(P.index)
    if len(missing):
        raise ValueError(f"{len(missing)} pooled runs have no probability row (e.g. {list(missing[:3])})")
    absent = sorted(set(table.stated) - set(P.columns))
    if absent:
        raise ValueError(f"no probability column for assessable label(s) {absent}")
    p = np.array([P.at[r, s] for r, s in zip(table.index, table.stated)], dtype=float)
    return 1.0 - p


def cl_scores(table: pd.DataFrame, P: pd.DataFrame) -> np.ndarray:
    """Confident Learning (Northcutt et al. 2021; PROTOCOLS.md v19 amendment 2026-10-26): per-class threshold t_j = mean
    P(j) over the pooled runs stated j; a run is an issue when the class with the highest P(j) among those with
    P(j) >= t_j differs from its stated label. Score = issue + (1 - P(stated)): issues first, each group by self-confidence."""
    base = flag_scores(table, P)
    X = P.loc[table.index].to_numpy(dtype=float)
    cols = list(P.columns)
    stated = table.stated.to_numpy()
    t = np.array([X[stated == c, j].mean() if (stated == c).any() else np.inf for j, c in enumerate(cols)])
    cand = np.where(X >= t[None, :], X, -np.inf)
    has = np.isfinite(cand).any(axis=1)
    assigned = np.array(cols, dtype=object)[cand.argmax(axis=1)]
    issue = has & (assigned != stated)
    return issue.astype(float) + base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--heldout", action="store_true", help="score the held-out set (test_metadata) instead of the dev folds")
    ap.add_argument("--plantings", type=Path, required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path template with {task}")
    ap.add_argument("--pair", action="append", default=[], help="a:b, a minus b")
    ap.add_argument("--cl", action="append", default=[], help="model name whose flags use the Confident Learning score")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    models = dict(m.split("=", 1) for m in a.model)
    tr = pd.read_csv(a.split_dir / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = None
    if a.heldout:
        meta = pd.read_csv(a.split_dir / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    else:
        meta = tr
        folds = pd.read_csv(a.split_dir / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
        if folds.isna().any():
            raise ValueError("training runs without a dev fold")
    files = sorted(glob.glob(str(a.plantings / a.pattern)))
    if not files:
        raise SystemExit(f"no plantings match {a.pattern} in {a.plantings}")
    plantings = read_plantings(files, a.split_dir, meta.index)
    score_rows, pair_rows = [], []
    for task in a.tasks.split(","):
        if a.heldout:
            seen = set(tr[task].dropna())
            assessable = lambda r, c: c in seen  # noqa: E731
        else:
            byfold = S.fit_classes_by_fold(tr[task], folds)
            assessable = lambda r, c: c in byfold[int(folds[r])]  # noqa: E731
        P = {m: read_probs(p, task, folds) for m, p in models.items()}
        built = [pool(meta, pl, task, assessable) for pl in plantings]
        pools, left_out = [b[0] for b in built], [b[1] for b in built]
        flags = {m: [(cl_scores if m in a.cl else flag_scores)(t, P[m]) for t in pools] for m in P}
        projects = np.unique(np.concatenate([t.project.to_numpy() for t in pools]))
        draws = S.project_draws(projects, f"flag|{task}", S.N_BOOT)
        draw_idx = []
        for t in pools:
            pos = {p: np.flatnonzero(t.project.to_numpy() == p) for p in projects}
            draw_idx.append([np.concatenate([pos[projects[i]] for i in d]) for d in draws])
        for m in P:
            rec = {b: [] for b in S.BUDGETS}
            cs, sat, auc, tooth5 = [], [], [], []
            att = [S.attainable_budgets(f, t.planted.to_numpy()) for t, f in zip(pools, flags[m])]
            for t, f, ab in zip(pools, flags[m], att):
                pl_ = t.planted.to_numpy()
                for b in S.BUDGETS:
                    rec[b].append(S.recall_if_attainable(f, pl_, b, ab))
                cs.append(S.choosing_score(f, pl_, ab) if ab else np.nan)
                sat.append(S.saturated_share(f, pl_))
                auc.append(roc_auc_score(pl_, f) if 0 < pl_.sum() < len(pl_) else np.nan)
                nt = (t.material != "tooth").to_numpy()
                tooth5.append(S.recall_if_attainable(f[nt], pl_[nt], 0.05, ab))
            b5, bcs = [], []
            for k in range(len(draws)):
                v5 = [S.recall_if_attainable(f[i[k]], t.planted.to_numpy()[i[k]], 0.05, ab)
                      for t, f, i, ab in zip(pools, flags[m], draw_idx, att)]
                b5.append(np.nanmean(v5) if np.isfinite(v5).any() else np.nan)
                v = [S.choosing_score(f[i[k]], t.planted.to_numpy()[i[k]], ab) if ab else np.nan
                     for t, f, i, ab in zip(pools, flags[m], draw_idx, att)]
                bcs.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
            lo5, hi5 = S.interval(b5)
            locs, hics = S.interval(bcs)
            mean_or_nan = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else np.nan  # noqa: E731
            score_rows.append({"task": task, "model": m, "n_plantings": len(pools),
                               "plantings_cant_check": int(sum(len(x) == 0 for x in att)),
                               "pool_mean": float(np.mean([len(t) for t in pools])),
                               "planted_mean": float(np.mean([t.planted.sum() for t in pools])),
                               "not_assessable_mean": float(np.mean(left_out)),
                               **{f"recall_at_{int(b * 100)}pct": mean_or_nan(rec[b]) for b in S.BUDGETS},
                               "recall_at_5pct_ci_low": lo5, "recall_at_5pct_ci_high": hi5,
                               "recall_at_5pct_draws": S.n_finite(b5),
                               "choosing_score": mean_or_nan(cs),
                               "choosing_ci_low": locs, "choosing_ci_high": hics, "choosing_draws": S.n_finite(bcs),
                               "roc_auc": mean_or_nan(auc), "saturated_share": mean_or_nan(sat),
                               "tooth_excluded_recall_at_5pct": mean_or_nan(tooth5)})
        for spec in a.pair:
            ma, mb = spec.split(":")
            common = [tuple(b for b in S.BUDGETS if b in S.attainable_budgets(fa, t.planted.to_numpy())
                            and b in S.attainable_budgets(fb, t.planted.to_numpy()))
                      for t, fa, fb in zip(pools, flags[ma], flags[mb])]
            usable = [k for k, c in enumerate(common) if c]
            if not usable:
                pair_rows.append({"task": task, "a": ma, "b": mb, "plantings_used": 0, "verdict": "can't check"})
                continue

            def diff(sel):
                vals = []
                for k in usable:
                    t, fa, fb, idx = pools[k], flags[ma][k], flags[mb][k], sel(k)
                    pl_ = t.planted.to_numpy()[idx]
                    vals.append(S.choosing_score(fa[idx], pl_, common[k]) - S.choosing_score(fb[idx], pl_, common[k]))
                return float(np.nanmean(vals))

            obs = diff(lambda k: np.arange(len(pools[k])))
            d = np.array([diff(lambda k, j=j: draw_idx[k][j]) for j in range(len(draws))])
            lo, hi = S.interval(d)
            fin = d[np.isfinite(d)]
            pair_rows.append({"task": task, "a": ma, "b": mb, "plantings_used": len(usable),
                              "budgets": ";".join(sorted({str(b) for k in usable for b in common[k]})),
                              "delta_choosing": obs, "ci_low": lo, "ci_high": hi, "draws_used": S.n_finite(d),
                              "frac_boot_positive": float((fin > 0).mean()) if len(fin) else np.nan,
                              "verdict": ma if lo > 0 else (mb if hi < 0 else "tie")})
        log.info("%s: %d models on %d plantings, pool %.0f runs, %.0f labelled runs not assessable",
                 task, len(P), len(pools), np.mean([len(t) for t in pools]), np.mean(left_out))
    pd.DataFrame(score_rows).to_csv(a.out / "flagging_scores.tsv", sep="\t", index=False)
    pd.DataFrame(pair_rows).to_csv(a.out / "flagging_paired.tsv", sep="\t", index=False)
    pd.set_option("display.width", 240)
    print(pd.DataFrame(score_rows).round(3).to_string(index=False))
    if pair_rows:
        print(pd.DataFrame(pair_rows).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
