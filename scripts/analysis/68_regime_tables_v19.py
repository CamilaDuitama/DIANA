#!/usr/bin/env python3
"""Dev-fold classification and mislabel detection by training-support regime (v19, C2d).

Regime of a class = its training-run count (few < 20 <= medium < 100 <= many).
Classification: macro-F1 over the eligible classes of the regime among the scored runs
(C2a), pooled out-of-fold, BioProject interval with the same resamples as
43_classification_v19.py (projects of the scored runs). Mislabel detection: for each
planting, the flag threshold is set on the whole task pool (the operating point a
curator would use: at most 5 % / 20 % of correctly labelled runs flagged); recall is then
the share of planted runs whose TRUE class is in the regime that are caught; averaged
over the plantings where the budget is attainable on the full data (decided once and
reused on every resample, as in 41_flagging_v19.py; NaN = can't check). Interval: the
resamples of 41_ (projects of the pools), threshold recomputed on each resample.
Pools, plantings and probabilities as in 41_flagging_v19.py. --train-fit scores models on the training runs they
were fitted on: every labelled training run, the dev plantings, no fold restriction.
--pair a:b adds paired differences a - b per regime on the same runs and BioProject draws
(macro-F1, and recall at 5 % over the plantings where both models attain 5 % on the full
data), written to <out>_paired.tsv; `*_draws` columns count the resamples that gave a value.
--pair-budgets 0.05,0.1,0.2 adds the same paired recall at 10 % and 20 % (columns recall10_*, recall20_*).

    ./env/bin/python scripts/analysis/68_regime_tables_v19.py --model LogReg=... --out <tsv>
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [r[0] for r in S.REGIMES]


def flagging_module():
    spec = importlib.util.spec_from_file_location("flag41", ROOT / "scripts/analysis/41_flagging_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def threshold(score: np.ndarray, planted: np.ndarray, budget: float) -> float:
    """Score threshold of the last ROC point with false-flag rate <= budget (as recall_at_budget)."""
    fpr, _, thr = roc_curve(planted, score, drop_intermediate=False)
    return float(thr[max(int(np.searchsorted(fpr, budget, side="right")) - 1, 0)])


def regime_recalls(pools, flags, regime_of, pool_draw, budget: float, with_draws: bool) -> dict:
    """Per regime, an array [planting, 1 + draws] of recall at `budget` (column 0 = full data):
    NaN where the budget is not attainable on the full planting (decided once) or no planted
    run of the regime is in the selection."""
    regs = {reg: [t.true.map(regime_of).eq(reg).to_numpy() for t in pools] for reg in REGIMES}
    n = 1 + (len(pool_draw[0]) if with_draws else 0)
    out = {reg: np.full((len(pools), n), np.nan) for reg in REGIMES}
    for k, (t, f) in enumerate(zip(pools, flags)):
        pl_full = t.planted.to_numpy()
        if budget not in S.attainable_budgets(f, pl_full):
            continue
        sels = [np.arange(len(t))] + (list(pool_draw[k]) if with_draws else [])
        for j, idx in enumerate(sels):
            sc, pl = f[idx], pl_full[idx]
            if not 0 < pl.sum() < len(pl):
                continue
            thr = threshold(sc, pl, budget)
            for reg in REGIMES:
                hit = pl & regs[reg][k][idx]
                if hit.any():
                    out[reg][k, j] = float((sc[hit] >= thr).mean())
    return out


def col_mean(m: np.ndarray, rows=None) -> np.ndarray:
    """Mean over plantings (rows) per column, NaN where no planting has a value."""
    m = m if rows is None else m[rows]
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(m).sum(0)
        return np.where(cnt > 0, np.nansum(m, 0) / np.maximum(cnt, 1), np.nan)


def tag(budget: float) -> str:
    return str(int(round(budget * 100)))


def flag_by_regime(pools, flags, regime_of, pool_draw, pair_budgets=(0.05,)) -> dict:
    """Per regime: recall at 5 % (point, interval, per-planting arrays) and 20 % (point), planted count;
    per-planting arrays with draws (`_r<pct>`) for every budget in `pair_budgets`."""
    r5 = regime_recalls(pools, flags, regime_of, pool_draw, 0.05, True)
    r20 = regime_recalls(pools, flags, regime_of, pool_draw, 0.20, False)
    extra = {b: regime_recalls(pools, flags, regime_of, pool_draw, b, True) for b in pair_budgets if b != 0.05}
    out = {}
    for reg in REGIMES:
        boot = col_mean(r5[reg])[1:]
        lo, hi = S.interval(boot)
        out[reg] = {"recall_5pct": float(col_mean(r5[reg])[0]), "recall_5pct_ci_low": lo, "recall_5pct_ci_high": hi,
                    "recall_5pct_draws": S.n_finite(boot), "recall_20pct": float(col_mean(r20[reg])[0]), "_r5": r5[reg],
                    "planted_in_regime_mean": float(np.mean([(t.planted.to_numpy() & t.true.map(regime_of).eq(reg).to_numpy()).sum()
                                                             for t in pools]))}
        out[reg].update({f"_r{tag(b)}": r[reg] for b, r in extra.items()})
    return out


def recall_contrib(pools, fa, fb, in_regime, budget: float) -> tuple[np.ndarray, list, int]:
    """Per-BioProject contributions to the full-data paired recall difference (mean over the plantings where both models
    attain `budget` and the regime has planted runs), with each model's threshold fixed on the full planting; swapping a
    BioProject's flags between the models flips the sign of its contribution (PROTOCOLS.md v19 amendment 2026-10-26)."""
    projects = sorted({p for t in pools for p in t.project})
    pos = {p: i for i, p in enumerate(projects)}
    total, used = np.zeros(len(projects)), 0
    for k, t in enumerate(pools):
        pl = t.planted.to_numpy()
        if budget not in S.attainable_budgets(fa[k], pl) or budget not in S.attainable_budgets(fb[k], pl):
            continue
        sel = pl & in_regime[k]
        if not sel.any() or not 0 < pl.sum() < len(pl):
            continue
        d = (fa[k][sel] >= threshold(fa[k], pl, budget)).astype(float) - (fb[k][sel] >= threshold(fb[k], pl, budget)).astype(float)
        np.add.at(total, [pos[p] for p in t.project.to_numpy()[sel]], d / sel.sum())
        used += 1
    return (total / used if used else total), projects, used


def recall_pair(ra: np.ndarray, rb: np.ndarray, t: str = "5") -> dict:
    """a - b over the plantings where both have a full-data value; interval over shared draws."""
    common = np.isfinite(ra[:, 0]) & np.isfinite(rb[:, 0])
    if not common.any():
        return {}
    d = col_mean(ra - rb, common)
    lo, hi = S.interval(d[1:])
    return {f"recall{t}_delta": float(d[0]), f"recall{t}_ci_low": lo, f"recall{t}_ci_high": hi, f"recall{t}_p_boot": S.bootstrap_p(d[1:]),
            f"recall{t}_plantings": int(common.sum()), f"recall{t}_draws": S.n_finite(d[1:])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split-dir", type=Path, default=ROOT / "data/splits_v19")
    ap.add_argument("--plantings", type=Path, default=ROOT / "results/planted_v19/dev")
    ap.add_argument("--pattern", default="planted_train_mixed_r0.1_s*.tsv")
    ap.add_argument("--model", action="append", required=True, help="name=probs template with {task} (flagging + classification)")
    ap.add_argument("--pred-only", action="append", default=[], help="name=prediction template with {task} (classification only)")
    ap.add_argument("--legacy-plantings-split", type=Path, default=None,
                    help="plantings made before provenance was recorded (v9): the split they were made on")
    ap.add_argument("--heldout", action="store_true", help="score the held-out set (test metadata and plantings)")
    ap.add_argument("--train-fit", action="store_true", help="score the training runs the models were fitted on")
    ap.add_argument("--pair", action="append", default=[], help="a:b, paired a minus b per regime")
    ap.add_argument("--whole-field", action="store_true",
                    help="one group, \"all\", holding every eligible class: whole-field values with the same resamples and pairs")
    ap.add_argument("--pair-budgets", default="0.05", help="comma-separated budgets for the paired recall (0.05 always included)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pair_budgets = sorted({0.05, *(float(b) for b in a.pair_budgets.split(","))})
    if a.whole_field:
        global REGIMES
        REGIMES = ["all"]
    if a.heldout and a.train_fit:
        raise SystemExit("--heldout and --train-fit are exclusive")
    F = flagging_module()
    tr = pd.read_csv(a.split_dir / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    folds = pd.read_csv(a.split_dir / "dev_folds.tsv", sep="\t").set_index("Run_accession").fold.reindex(tr.index)
    ev = (pd.read_csv(a.split_dir / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
          if a.heldout else tr)
    plantings = F.read_plantings(sorted(glob.glob(str(a.plantings / a.pattern))), a.split_dir, ev.index,
                                 legacy_split=a.legacy_plantings_split)
    models = dict(m.split("=", 1) for m in a.model)
    preds = dict(m.split("=", 1) for m in a.pred_only)
    rows, pair_rows = [], []
    for task in TASKS:
        elig = S.eligible_classes(a.split_dir / "class_eligibility.tsv", task)
        by_reg = {"all": set(elig)} if a.whole_field else S.regime_classes(tr[task], elig)
        regime_of = {c: r for r, cs in by_reg.items() for c in cs}
        counts = tr[task].dropna().value_counts()
        if a.heldout or a.train_fit:
            seen = set(tr[task].dropna())
            scored = S.scored_heldout(ev[task], seen)
            assessable = lambda r, c: c in seen  # noqa: E731
        else:
            scored = S.scored_devfold(tr[task], folds)
            byfold = S.fit_classes_by_fold(tr[task], folds)
            assessable = lambda r, c: c in byfold[int(folds[r])]  # noqa: E731
        y = ev[task]
        pools = [F.pool(ev, pl, task, assessable)[0] for pl in plantings]
        cls_runs = scored[scored].index
        g_cls = ev.loc[cls_runs, "archive_project"].astype(str).to_numpy()
        # mislabel resamples exactly as 41_: projects present in the pools, key flag|<task>
        projects = np.unique(np.concatenate([t.project.to_numpy() for t in pools]))
        draws = S.project_draws(projects, f"flag|{task}", S.N_BOOT)
        pool_pos = [{p: np.flatnonzero(t.project.to_numpy() == p) for p in projects} for t in pools]
        pool_draw = [[np.concatenate([pp[projects[i]] for i in d]) for d in draws] for pp in pool_pos]
        sources = {**{m: ("probs", p) for m, p in models.items()}, **{m: ("preds", p) for m, p in preds.items()}}
        kept_pred, kept_flag, kept_raw = {}, {}, {}
        for m, (kind, tpl) in sources.items():
            if kind == "probs":
                P = F.read_probs(tpl, task, None if a.heldout or a.train_fit else folds)
                pred = P.idxmax(axis=1)
            else:
                d = pd.read_csv(tpl.format(task=task), sep="\t", dtype=str).set_index("Run_accession")
                pred, P = d.y_pred, None
            yy = y.loc[cls_runs].astype(str).to_numpy()
            missing = cls_runs.difference(pred.dropna().index)
            if len(missing):
                raise ValueError(f"{m}/{task}: {len(missing)} scored runs have no prediction")
            pp = pred.loc[cls_runs].astype(str).to_numpy()
            flags = [F.flag_scores(t, P) for t in pools] if P is not None else None
            flag_res = flag_by_regime(pools, flags, regime_of, pool_draw, pair_budgets) if flags is not None else None
            kept_pred[m], kept_flag[m], kept_raw[m] = pp, flag_res, flags
            for reg in REGIMES:
                cls = by_reg[reg] & set(yy)
                if not cls:
                    continue
                sw = S.score_with_interval(yy, pp, g_cls, cls, f"cls|{task}|{reg}")  # same resamples as 43_
                row = {"task": task, "regime": reg, "model": m, "n_classes": len(cls), "classes": ";".join(sorted(cls)),
                       "macro_f1": sw["f1_macro_eligible"], "f1_ci_low": sw["ci_low"], "f1_ci_high": sw["ci_high"]}
                if flag_res is not None:
                    row.update({k: v for k, v in flag_res[reg].items() if not k.startswith("_")})
                rows.append(row)
        yy = y.loc[cls_runs].astype(str).to_numpy()
        for spec in a.pair:
            ma, mb = spec.split(":")
            for reg in REGIMES:
                cls = by_reg[reg] & set(yy)
                if not cls:
                    continue
                r = S.paired_difference(cls_runs, cls_runs, yy, kept_pred[ma], kept_pred[mb], g_cls, cls,
                                        f"regpair|{task}|{reg}")
                row = {"task": task, "regime": reg, "a": ma, "b": mb, "f1_delta": r["delta"], "f1_ci_low": r["ci_low"],
                       "f1_ci_high": r["ci_high"], "f1_verdict": {"a": ma, "b": mb}.get(r["verdict"], "tie")}
                fa, fb = kept_flag[ma], kept_flag[mb]
                for b in (pair_budgets if fa is not None and fb is not None else []):
                    t = tag(b)
                    rp = recall_pair(fa[reg][f"_r{t}"], fb[reg][f"_r{t}"], t)
                    if rp:
                        lo, hi = rp[f"recall{t}_ci_low"], rp[f"recall{t}_ci_high"]
                        row.update({**rp, f"recall{t}_verdict": ma if lo > 0 else (mb if hi < 0 else "tie")})
                        if t == "5":
                            in_reg = [tt.true.map(regime_of).eq(reg).to_numpy() for tt in pools]
                            c, _, used = recall_contrib(pools, kept_raw[ma], kept_raw[mb], in_reg, b)
                            if used:
                                row["recall5_p_perm"] = S.sign_flip_p(c, f"permflag|{task}|{reg}|{ma}|{mb}")
                                if abs(c.sum() - rp["recall5_delta"]) > 1e-9:
                                    raise ValueError(f"{task}/{reg}: contributions do not add up to the recall difference")
                pair_rows.append(row)
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(a.out.with_name(a.out.stem + "_paired.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["classes"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
