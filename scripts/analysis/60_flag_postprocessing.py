#!/usr/bin/env python3
"""X11 (Phase 3): turning the out-of-fold predictions into better flags, steps 3.1 to 3.7.

Every step is applied to the frozen reference (5-seed average) and to the tuned logistic
regression, on one benchmark (single-label or whole-record plantings), and judged with the
choosing score (mean catch rate over the attainable budgets among 5 %, 10 % and 20 %) and
recall at 5 %, paired over BioProjects against the same model without the step (1,000
shared project draws). All settings are fixed in PROJECT.md §8 (X11 protocol).

  3.1  group median: score minus the median of the run's (study, label on file) group, groups >= 5
  3.2  can't check: labels of classes recognised < 20 % or carried by < 2 training studies -> score 0
  3.3  temperature scaling fitted on the four training folds, applied to the fifth
  3.4  the half-and-half average of the two models' probabilities (a third "model")
  3.5  unusual sample (61_unusual_sample_score.py) -> score 0
  3.6  unseen label combination (pairs never seen in the training folds) -> score + 1
  3.7  one flag per run from the four tasks (max, mean); whole-record benchmark only

Outputs results/flag_postprocessing_v9/<bench>/<task>/{scores,paired,report}_<step>.tsv and
scores_base.tsv; --pool gathers every paired_*.tsv into paired_all.tsv and prints the verdicts.

    ./env/bin/python scripts/analysis/60_flag_postprocessing.py --bench single --task material --steps 3.1,3.4
    ./env/bin/python scripts/analysis/60_flag_postprocessing.py --bench record --steps 3.7
    ./env/bin/python scripts/analysis/60_flag_postprocessing.py --pool
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
F = import_module("59_devfold_flagging_score")

logger = logging.getLogger(__name__)
ROOT, SPLITS, TASKS, BUDGETS, BUDGET = F.ROOT, F.SPLITS, F.TASKS, F.BUDGETS, F.BUDGET
OUT_ROOT = ROOT / "results/flag_postprocessing_v9"
PATTERNS = {"single": "planted_train_mixed_r0.1_s*.tsv", "record": "planted_train_record_r0.1_s*.tsv"}
MODELS = ["reference", "LogisticRegression_Bal"]
STEPS = ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"]
MIN_GROUP, STUDY_FLAG_SHARE = 5, 0.5        # 3.1
MIN_RECALL, MIN_STUDIES = 0.20, 2           # 3.2
T_BOUNDS, EPS = (0.05, 20.0), 1e-8          # 3.3
COMBO_BUMP = 1.0                            # 3.6
N_BOOT = 1000


# ----------------------------------------------------------------------------- data
def load_probs(task: str) -> dict[str, pd.DataFrame]:
    return {"reference": F.reference_probs(task), "LogisticRegression_Bal": F.baseline_probs("LogisticRegression_Bal", task)}


def base_table(P: pd.DataFrame, pl: pd.DataFrame, task: str, meta: pd.DataFrame, folds: pd.Series) -> pd.DataFrame:
    """One row per scorable run of one planting: base flag score, planted flags, study, fold, labels."""
    had_label = meta.loc[pl.index, task].notna().to_numpy()
    pl_t = pl[pl[task].notna() & pl[f"{task}_planted"].notna() & had_label]
    s = F.flag_scores(P, pl_t[task]).dropna()
    idx = s.index
    rec = pl["record_planted"] if "record_planted" in pl.columns else pl[f"{task}_planted"]
    return pd.DataFrame({"score": s.to_numpy(dtype=float), "planted": pl_t.loc[idx, f"{task}_planted"].astype(bool).to_numpy(),
                         "record_planted": rec.loc[idx].astype(bool).to_numpy(),
                         "project": meta.loc[idx, "archive_project"].astype(str).to_numpy(),
                         "label": pl_t.loc[idx, task].astype(str).to_numpy(), "true": meta.loc[idx, task].astype(str).to_numpy(),
                         "fold": folds.loc[idx].to_numpy(), "material_on_file": pl.loc[idx, "material"].astype(object).to_numpy(),
                         "kind": "likely mistake"}, index=idx)


# ----------------------------------------------------------------------------- scoring
def roc_metrics(score: np.ndarray, planted: np.ndarray) -> tuple[dict, float]:
    """Recall at every budget (the held-out definition: last ROC point with FPR <= budget) and the
    share of clean runs at the maximum score, from one ROC curve."""
    if planted.sum() == 0 or (~planted).sum() == 0:
        return {b: np.nan for b in BUDGETS}, np.nan
    fpr, tpr, _ = roc_curve(planted, score)
    rec = {b: float(tpr[max(int(np.searchsorted(fpr, b, side="right") - 1), 0)]) for b in BUDGETS}
    clean = score[~planted]
    return rec, float((clean >= np.nanmax(score) - 1e-9).mean())


def choosing(rec: dict, sat: float) -> tuple[float, int]:
    vals = [rec[b] for b in BUDGETS if sat <= b]
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


def summarise(per: list[pd.DataFrame]) -> dict:
    acc = {f"recall_at_{int(b * 100)}pct": [] for b in BUDGETS}
    acc.update({"choosing_score": [], "budgets_attainable": [], "saturated_share": []})
    for t in per:
        rec, sat = roc_metrics(t.score.to_numpy(), t.planted.to_numpy())
        for b in BUDGETS:
            acc[f"recall_at_{int(b * 100)}pct"].append(rec[b])
        cs, nb = choosing(rec, sat)
        acc["choosing_score"].append(cs); acc["budgets_attainable"].append(nb); acc["saturated_share"].append(sat)
    out = {k: (float(np.min(v)) if k == "budgets_attainable" else float(np.nanmean(v)) if np.isfinite(v).any() else np.nan) for k, v in acc.items()}
    out["runs_scored_mean"] = float(np.mean([len(t) for t in per])); out["planted_mean"] = float(np.mean([t.planted.sum() for t in per]))
    out["marked_mean"] = float(np.mean([(t.kind != "likely mistake").sum() for t in per]))
    out["planted_marked_mean"] = float(np.mean([((t.kind != "likely mistake") & t.planted).sum() for t in per]))
    out["n_plantings"] = len(per)
    return out


def bootstrap(per: list[pd.DataFrame], projects: np.ndarray, draws: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per draw: recall at 5 % and the choosing score averaged over plantings; projects resampled, runs kept."""
    codes = [pd.Categorical(t.project, categories=projects).codes for t in per]
    scores = [t.score.to_numpy() for t in per]; planted = [t.planted.to_numpy() for t in per]
    r5, cs = np.full(len(draws), np.nan), np.full(len(draws), np.nan)
    for d, draw in enumerate(draws):
        counts = np.bincount(draw, minlength=len(projects))
        recs, css = [], []
        for c, s, p in zip(codes, scores, planted):
            idx = np.repeat(np.arange(len(s)), counts[c])
            if len(idx) == 0:
                continue
            rec, sat = roc_metrics(s[idx], p[idx])
            recs.append(rec[BUDGET]); css.append(choosing(rec, sat)[0])
        if recs:
            r5[d] = np.nanmean(recs) if np.isfinite(recs).any() else np.nan
            cs[d] = np.nanmean(css) if np.isfinite(css).any() else np.nan
    return r5, cs


def ci(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def pair(a: str, b: str, summ: dict, boots: dict) -> dict:
    """a minus b on the choosing score (and recall at 5 %), on the shared draws."""
    d = boots[a][1] - boots[b][1]; d5 = boots[a][0] - boots[b][0]
    obs = summ[a]["choosing_score"] - summ[b]["choosing_score"]
    lo, hi = ci(d); lo5, hi5 = ci(d5)
    verdict = ("can't check" if not np.isfinite(obs) or not np.isfinite(lo) else "tie" if lo <= 0 <= hi else a if obs > 0 else b)
    return {"a": a, "b": b, "a_choosing": summ[a]["choosing_score"], "b_choosing": summ[b]["choosing_score"], "delta_choosing": obs,
            "ci_low": lo, "ci_high": hi, "frac_boot_positive": float(np.nanmean(d > 0)) if np.isfinite(d).any() else np.nan, "verdict": verdict,
            "a_recall_at_5pct": summ[a]["recall_at_5pct"], "b_recall_at_5pct": summ[b]["recall_at_5pct"],
            "delta_recall_at_5pct": summ[a]["recall_at_5pct"] - summ[b]["recall_at_5pct"], "ci_low_5pct": lo5, "ci_high_5pct": hi5}


def score_and_write(tables: dict[str, list], pairs: list[tuple[str, str]], projects, draws, out_dir: Path, step: str, extra_cols: dict):
    summ = {n: summarise(per) for n, per in tables.items()}
    boots = {n: bootstrap(per, projects, draws) for n, per in tables.items()}
    rows = []
    for n, sm in summ.items():
        lo, hi = ci(boots[n][0]); clo, chi = ci(boots[n][1])
        rows.append({**extra_cols, "step": step, "model": n, **sm, "ci_low_5pct": lo, "ci_high_5pct": hi, "choosing_ci_low": clo, "choosing_ci_high": chi,
                     "cant_check_5pct": sm["saturated_share"] > BUDGET})
    pd.DataFrame(rows).to_csv(out_dir / f"scores_{step}.tsv", sep="\t", index=False)
    pr = pd.DataFrame([{**extra_cols, "step": step, **pair(a, b, summ, boots)} for a, b in pairs])
    if len(pr):
        pr.to_csv(out_dir / f"paired_{step}.tsv", sep="\t", index=False)
    return pd.DataFrame(rows), pr


# ----------------------------------------------------------------------------- steps
def step_group_median(tables: dict[str, list]) -> tuple[dict, pd.DataFrame]:
    out, rep = {}, []
    for m, per in tables.items():
        new = []
        for pi, t in enumerate(per):
            g = t.groupby(["project", "label"]).score
            med, size = g.transform("median").to_numpy(), g.transform("size").to_numpy()
            u = t.copy(); u["score"] = np.where(size >= MIN_GROUP, t.score.to_numpy() - med, t.score.to_numpy()); new.append(u)
            _, thr = F.recall_at_budget(t.score.to_numpy(), t.planted.to_numpy())
            st = t.assign(flag=t.score >= thr).groupby("project").flag.agg(["mean", "size"])
            flagged = st[(st["size"] >= MIN_GROUP) & (st["mean"] > STUDY_FLAG_SHARE)].index.tolist()
            planted_share = t.groupby(["project", "label"]).planted.transform("mean").to_numpy(); pl_ = t.planted.to_numpy()
            rep.append({"model": m, "planting": pi, "study_level_flags": len(flagged), "studies": ";".join(flagged),
                        "share_planted_in_small_groups": float((pl_ & (size < MIN_GROUP)).sum() / max(pl_.sum(), 1)),
                        # several runs of one study planted with the same wrong label form a group of their own; the median
                        # then cancels them: reported so the reader sees where the step fails
                        "share_planted_in_planted_majority_groups": float((pl_ & (size >= MIN_GROUP) & (planted_share > 0.5)).sum() / max(pl_.sum(), 1)),
                        "share_runs_in_groups": float((size >= MIN_GROUP).mean())})
        out[f"{m}+3.1"] = new
    rep = pd.DataFrame(rep)
    agg = rep.groupby("model").agg(study_level_flags_mean=("study_level_flags", "mean"), share_planted_in_small_groups=("share_planted_in_small_groups", "mean"),
                                   share_planted_in_planted_majority_groups=("share_planted_in_planted_majority_groups", "mean"),
                                   share_runs_in_groups=("share_runs_in_groups", "mean")).reset_index()
    n_pl = rep.planting.nunique()
    agg["studies_flagged_in_half_of_plantings"] = [";".join(sorted(s for s, c in pd.Series(sum((x.split(";") for x in rep[rep.model == m].studies if x), [])).value_counts().items() if c >= n_pl / 2)) for m in agg.model]
    return out, agg


def step_cant_check(tables: dict[str, list], preds: dict[str, pd.Series]) -> tuple[dict, pd.DataFrame]:
    out, rep = {}, []
    for m, per in tables.items():
        new = []
        for pi, t in enumerate(per):
            u = t.copy(); marked_classes = {}
            for k in sorted(t.fold.unique()):
                tr = t[t.fold != k]
                hit = (preds[m].reindex(tr.index).astype(str).to_numpy() == tr.label.to_numpy())
                recall = pd.Series(hit, index=tr.index).groupby(tr.label.to_numpy()).mean()
                studies = tr.groupby("label").project.nunique()
                bad = set(recall[recall < MIN_RECALL].index) | set(studies[studies < MIN_STUDIES].index)
                mask = (u.fold == k) & u.label.isin(bad)
                u.loc[mask, "score"] = 0.0; u.loc[mask, "kind"] = "can't check"; marked_classes[k] = sorted(bad)
            new.append(u)
            rep.append({"model": m, "planting": pi, "runs_marked": int((u.kind == "can't check").sum()),
                        "planted_marked": int(((u.kind == "can't check") & u.planted).sum()), "planted_total": int(u.planted.sum()),
                        "classes_marked": ";".join(f"fold{k}:{','.join(v)}" for k, v in marked_classes.items())})
        out[f"{m}+3.2"] = new
    rep = pd.DataFrame(rep)
    agg = rep.groupby("model").agg(runs_marked_mean=("runs_marked", "mean"), planted_marked_mean=("planted_marked", "mean"), planted_total_mean=("planted_total", "mean")).reset_index()
    agg["classes_marked_planting0"] = [rep[(rep.model == m) & (rep.planting == 0)].classes_marked.iloc[0] for m in agg.model]
    return out, agg


def fit_temperature(logp: np.ndarray, y: np.ndarray) -> float:
    def nll(T):
        return -float(log_softmax(logp / T, axis=1)[np.arange(len(y)), y].mean())
    return float(minimize_scalar(nll, bounds=T_BOUNDS, method="bounded").x)


def step_temperature(probs: dict[str, pd.DataFrame], plants: list[pd.DataFrame], task: str, meta, folds) -> tuple[dict, pd.DataFrame]:
    out, rep = {}, []
    for m, P in probs.items():
        na = P.isna(); logp_all = np.log(np.clip(P.fillna(0.0).to_numpy(dtype=float), EPS, 1.0))
        cols = list(P.columns); col_idx = {c: i for i, c in enumerate(cols)}
        fold_of = folds.reindex(P.index).to_numpy()
        new = []
        for pi, pl in enumerate(plants):
            on_file = pl[task].reindex(P.index).astype(object)
            y = np.array([col_idx.get(str(v), -1) if pd.notna(v) else -1 for v in on_file])
            Pc = np.full_like(logp_all, np.nan); Ts = {}
            for k in sorted(set(fold_of)):
                fit = (fold_of != k) & (y >= 0) & ~na.to_numpy().any(axis=1)
                T = fit_temperature(logp_all[fit], y[fit]); Ts[k] = T
                ev = fold_of == k
                Pc[ev] = softmax(logp_all[ev] / T, axis=1)
            Pc = pd.DataFrame(Pc, index=P.index, columns=cols).mask(na)
            new.append(base_table(Pc, pl, task, meta, folds))
            rep.append({"model": m, "planting": pi, **{f"T_fold{k}": v for k, v in Ts.items()}})
        out[f"{m}+3.3"] = new
    rep = pd.DataFrame(rep)
    return out, rep.groupby("model").mean(numeric_only=True).drop(columns="planting").reset_index()


def step_average(probs: dict[str, pd.DataFrame], plants, task, meta, folds) -> tuple[dict, pd.DataFrame]:
    a, b = probs["reference"], probs["LogisticRegression_Bal"]
    cols = sorted(set(a.columns) | set(b.columns)); idx = a.index.intersection(b.index)
    Pavg = (a.reindex(index=idx, columns=cols).fillna(0.0) + b.reindex(index=idx, columns=cols).fillna(0.0)) / 2.0
    per = [base_table(Pavg, pl, task, meta, folds) for pl in plants]
    rep = pd.DataFrame([{"model": "average", "runs_in_both": len(idx), "classes_union": len(cols),
                         "classes_reference_only": len(set(a.columns) - set(b.columns)), "classes_logistic_only": len(set(b.columns) - set(a.columns))}])
    return {"average": per}, rep


def step_unusual(tables: dict[str, list]) -> tuple[dict, pd.DataFrame]:
    path = OUT_ROOT / "unusual_scores.tsv"
    if not path.exists():
        raise SystemExit(f"{path} missing: run 61_unusual_sample_score.py first")
    un = pd.read_csv(path, sep="\t").set_index("Run_accession")
    out, rep = {}, []
    for m, per in tables.items():
        new = []
        for pi, t in enumerate(per):
            u = t.copy(); mask = un.unusual.reindex(u.index).fillna(False).astype(bool).to_numpy()
            u.loc[mask, "score"] = 0.0; u.loc[mask, "kind"] = "unusual sample"; new.append(u)
            rep.append({"model": m, "planting": pi, "runs_marked": int(mask.sum()), "planted_marked": int((mask & u.planted.to_numpy()).sum()),
                        "planted_total": int(u.planted.sum()), "prjeb42014_marked": int((mask & (u.project == "PRJEB42014").to_numpy()).sum()),
                        "prjeb42014_runs": int((u.project == "PRJEB42014").sum())})
        out[f"{m}+3.5"] = new
    rep = pd.DataFrame(rep)
    return out, rep.groupby("model").mean(numeric_only=True).drop(columns="planting").reset_index()


def step_combinations(tables: dict[str, list], plants: list[pd.DataFrame], task: str, folds: pd.Series) -> tuple[dict, pd.DataFrame]:
    others = [u for u in TASKS if u != task]
    out, rep = {}, []
    for m, per in tables.items():
        new = []
        for pi, t in enumerate(per):
            pl = plants[pi]; lab = pl[TASKS].astype(object); fold_of = folds.reindex(pl.index)
            unseen = pd.Series(False, index=t.index)
            for k in sorted(t.fold.unique()):
                tr = lab[(fold_of != k).to_numpy() & lab[task].notna().to_numpy()]
                seen = {u: set(zip(tr[task].astype(str), tr[u].astype(str))) for u in others}
                ev = t.index[t.fold == k]; L = lab.loc[ev]
                for u in others:
                    has = L[u].notna().to_numpy()
                    pairs_ = list(zip(L[task].astype(str), L[u].astype(str)))
                    unseen.loc[ev] |= has & np.array([p not in seen[u] for p in pairs_])
            u_ = t.copy(); u_["score"] = t.score.to_numpy() + COMBO_BUMP * unseen.to_numpy(); new.append(u_)
            rep.append({"model": m, "planting": pi, "share_clean_unseen": float(unseen[~t.planted].mean()), "share_planted_unseen": float(unseen[t.planted].mean()) if t.planted.any() else np.nan})
        out[f"{m}+3.6"] = new
    rep = pd.DataFrame(rep)
    return out, rep.groupby("model").mean(numeric_only=True).drop(columns="planting").reset_index()


def step_combined_tasks(per_task: dict[str, dict[str, list]], model: str) -> dict[str, list]:
    """Whole-record benchmark: one score per run from its scorable tasks (max and mean), planted = record swap;
    the single-task scores are re-expressed against the same planted definition."""
    out = {f"{model}+3.7max": [], f"{model}+3.7mean": []}
    for task in TASKS:
        out[f"{model} {task}"] = []
    n_pl = len(next(iter(per_task.values()))[model])
    for pi in range(n_pl):
        parts = []
        for task in TASKS:
            t = per_task[task][model][pi].copy(); t["planted"] = t.record_planted
            out[f"{model} {task}"].append(t); parts.append(t.score.rename(task))
        S = pd.concat(parts, axis=1)
        base = pd.concat([per_task[task][model][pi][["record_planted", "project", "fold"]] for task in TASKS]).groupby(level=0).first()
        for name, agg in (("max", S.max(axis=1)), ("mean", S.mean(axis=1))):
            out[f"{model}+3.7{name}"].append(pd.DataFrame({"score": agg.loc[base.index].to_numpy(), "planted": base.record_planted.to_numpy(),
                                                            "record_planted": base.record_planted.to_numpy(), "project": base.project.to_numpy(),
                                                            "label": "", "true": "", "fold": base.fold.to_numpy(), "kind": "likely mistake"}, index=base.index))
    return out


# ----------------------------------------------------------------------------- 3.8
COMPOSE_ORDER = ["3.3", "3.1", "3.6", "3.2", "3.5"]      # calibration, group median, combination bump, then the masks
KINDS = ["can't check", "unusual sample", "study-level", "convention conflict", "likely mistake"]


def strip_suffix(tables: dict[str, list]) -> dict[str, list]:
    return {n.split("+")[0]: per for n, per in tables.items()}


def apply_step(step: str, name: str, tables: list, P: pd.DataFrame, plants, task, meta, folds) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """One step on one named model; 3.3 recomputes the tables from calibrated probabilities (and returns them)."""
    if step == "3.3":
        new, rep = step_temperature({name: P}, plants, task, meta, folds)
        return new[f"{name}+3.3"], rep, P      # the tables already carry the calibrated scores; P stays for 3.2's argmax (unchanged by T)
    if step == "3.1":
        new, rep = step_group_median({name: tables})
    elif step == "3.2":
        new, rep = step_cant_check({name: tables}, {name: P.idxmax(axis=1)})
    elif step == "3.5":
        new, rep = step_unusual({name: tables})
    elif step == "3.6":
        new, rep = step_combinations({name: tables}, plants, task, folds)
    else:
        raise ValueError(step)
    return new[f"{name}+{step}"], rep, P


def flag_kinds(per: list[pd.DataFrame]) -> pd.DataFrame:
    """Every flag at the 5 % threshold sorted into one kind, precedence as in the protocol; mean over plantings."""
    rows = []
    for t in per:
        _, thr = F.recall_at_budget(t.score.to_numpy(), t.planted.to_numpy())
        flagged = t.score >= thr
        st = t.assign(flag=flagged).groupby("project").flag.agg(["mean", "size"])
        study_level = t.project.isin(st[(st["size"] >= MIN_GROUP) & (st["mean"] > STUDY_FLAG_SHARE)].index)
        kind = t.kind.copy()
        kind[(kind == "likely mistake") & study_level] = "study-level"
        kind[(kind == "likely mistake") & (t.material_on_file.astype(str) == "tooth") & flagged] = "convention conflict"
        for k in KINDS:
            m = kind == k
            rows.append({"kind": k, "runs": int(m.sum()), "flagged": int((m & flagged).sum()), "planted_flagged": int((m & flagged & t.planted).sum()),
                         "planted_total": int((m & t.planted).sum())})
    return pd.DataFrame(rows).groupby("kind").mean().reindex(KINDS).reset_index()


def run_compose(bench: str, task: str, specs: dict[str, list[str]], plants, meta, folds, n_boot: int) -> None:
    """specs: {"reference": ["3.3", "3.1"], "average": ["3.2"], ...}; the combination is paired against the unmodified
    model and against every one of its single steps (recomputed here, same draws)."""
    global N_BOOT
    N_BOOT = n_boot
    out_dir = OUT_ROOT / bench / task; out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"bench": bench, "task": task}
    probs = load_probs(task)
    if "average" in specs:
        a, b = probs["reference"], probs["LogisticRegression_Bal"]
        cols = sorted(set(a.columns) | set(b.columns)); idx = a.index.intersection(b.index)
        probs["average"] = (a.reindex(index=idx, columns=cols).fillna(0.0) + b.reindex(index=idx, columns=cols).fillna(0.0)) / 2.0
    projects = np.array(sorted(meta.loc[meta[task].notna(), "archive_project"].astype(str).unique()))
    draws = draws_for(bench, task, projects)
    tables, pairs, reports, kinds = {}, [], [], []
    for name, steps in specs.items():
        P = probs[name]; base = [base_table(P, pl, task, meta, folds) for pl in plants]
        tables[name] = base
        singles = []
        for step in steps:
            single, _, _ = apply_step(step, name, base, P, plants, task, meta, folds)
            tables[f"{name}+{step}"] = single; singles.append(f"{name}+{step}")
        cur = base
        for step in [s_ for s_ in COMPOSE_ORDER if s_ in steps]:
            cur, rep, P = apply_step(step, name, cur, P, plants, task, meta, folds)
            reports.append(rep.assign(model=name, step=step))
        combo = f"{name}+3.8[{','.join(steps)}]"
        tables[combo] = cur
        pairs += [(combo, name)] + [(combo, s_) for s_ in singles]
        if name == "average":
            pairs += [(combo, m) for m in MODELS]
        kinds.append(flag_kinds(cur).assign(model=combo))
    sc, pr = score_and_write(tables, pairs, projects, draws, out_dir, "3.8", extra)
    pd.concat(reports, ignore_index=True).to_csv(out_dir / "report_3.8.tsv", sep="	", index=False)
    kd = pd.concat(kinds, ignore_index=True); kd.to_csv(out_dir / "kinds_3.8.tsv", sep="	", index=False)
    pd.set_option("display.width", 250)
    print(f"\n=== {bench} / {task} / step 3.8 ===")
    print(sc[["model", "planted_mean", "marked_mean", "planted_marked_mean", "recall_at_5pct", "ci_low_5pct", "ci_high_5pct", "recall_at_10pct", "recall_at_20pct",
              "choosing_score", "choosing_ci_low", "choosing_ci_high", "budgets_attainable"]].round(3).to_string(index=False))
    print(pr[["a", "b", "delta_choosing", "ci_low", "ci_high", "frac_boot_positive", "verdict", "delta_recall_at_5pct"]].round(3).to_string(index=False))
    print(kd.round(2).to_string(index=False))


# ----------------------------------------------------------------------------- driver
def draws_for(bench: str, key: str, projects: np.ndarray) -> list[np.ndarray]:
    rng = F.rng_for(bench, key)
    return [rng.integers(0, len(projects), size=len(projects)) for _ in range(N_BOOT)]


def run_task(bench: str, task: str, steps: list[str], plants: list[pd.DataFrame], meta, folds, n_boot: int) -> None:
    global N_BOOT
    N_BOOT = n_boot
    out_dir = OUT_ROOT / bench / task; out_dir.mkdir(parents=True, exist_ok=True)
    extra = {"bench": bench, "task": task}
    probs = load_probs(task)
    preds = {m: P.idxmax(axis=1) for m, P in probs.items()}
    tables = {m: [base_table(P, pl, task, meta, folds) for pl in plants] for m, P in probs.items()}
    projects = np.array(sorted(meta.loc[meta[task].notna(), "archive_project"].astype(str).unique()))
    draws = draws_for(bench, task, projects)
    t0 = time.time()
    base_scores, _ = score_and_write(tables, [], projects, draws, out_dir, "base", extra)
    logger.info("%s/%s base scored in %.0f s", bench, task, time.time() - t0)
    for step in steps:
        t1 = time.time()
        if step == "3.1":
            new, rep = step_group_median(tables)
        elif step == "3.2":
            new, rep = step_cant_check(tables, preds)
        elif step == "3.3":
            new, rep = step_temperature(probs, plants, task, meta, folds)
        elif step == "3.4":
            new, rep = step_average(probs, plants, task, meta, folds)
        elif step == "3.5":
            new, rep = step_unusual(tables)
        elif step == "3.6":
            new, rep = step_combinations(tables, plants, task, folds)
        else:
            continue
        pairs = [("average", m) for m in MODELS] if step == "3.4" else [(n, n.split("+")[0]) for n in new]
        sc, pr = score_and_write({**tables, **new}, pairs, projects, draws, out_dir, step, extra)
        rep.to_csv(out_dir / f"report_{step}.tsv", sep="\t", index=False)
        pd.set_option("display.width", 250)
        print(f"\n=== {bench} / {task} / step {step} ({time.time() - t1:.0f} s) ===")
        print(sc[["model", "planted_mean", "runs_scored_mean", "marked_mean", "planted_marked_mean", "recall_at_5pct", "ci_low_5pct", "ci_high_5pct",
                  "recall_at_10pct", "recall_at_20pct", "choosing_score", "choosing_ci_low", "choosing_ci_high", "budgets_attainable", "saturated_share"]].round(3).to_string(index=False))
        print(pr[["a", "b", "delta_choosing", "ci_low", "ci_high", "frac_boot_positive", "verdict", "delta_recall_at_5pct", "ci_low_5pct", "ci_high_5pct"]].round(3).to_string(index=False))
        print(rep.round(3).to_string(index=False))


def run_combined(bench: str, plants, meta, folds, n_boot: int) -> None:
    global N_BOOT
    N_BOOT = n_boot
    out_dir = OUT_ROOT / bench / "all_tasks"; out_dir.mkdir(parents=True, exist_ok=True)
    per_task = {task: {m: [base_table(P, pl, task, meta, folds) for pl in plants] for m, P in load_probs(task).items()} for task in TASKS}
    projects = np.array(sorted(meta["archive_project"].astype(str).unique()))
    draws = draws_for(bench, "all_tasks", projects)
    tables, pairs = {}, []
    for m in MODELS:
        tb = step_combined_tasks(per_task, m); tables.update(tb)
        singles = [f"{m} {task}" for task in TASKS]
        pairs += [(f"{m}+3.7{agg}", s) for agg in ("max", "mean") for s in singles] + [(f"{m}+3.7max", f"{m}+3.7mean")]
    sc, pr = score_and_write(tables, pairs, projects, draws, out_dir, "3.7", {"bench": bench, "task": "all_tasks"})
    pd.set_option("display.width", 250)
    print(f"\n=== {bench} / all tasks / step 3.7 (planted = record swap) ===")
    print(sc[["model", "planted_mean", "runs_scored_mean", "recall_at_5pct", "ci_low_5pct", "ci_high_5pct", "recall_at_10pct", "recall_at_20pct",
              "choosing_score", "choosing_ci_low", "choosing_ci_high", "budgets_attainable", "saturated_share"]].round(3).to_string(index=False))
    print(pr[["a", "b", "delta_choosing", "ci_low", "ci_high", "frac_boot_positive", "verdict", "delta_recall_at_5pct"]].round(3).to_string(index=False))


def pool() -> None:
    files = sorted(glob.glob(str(OUT_ROOT / "*" / "*" / "paired_*.tsv")))
    pr = pd.concat([pd.read_csv(f, sep="\t", dtype={"step": str}) for f in files], ignore_index=True)
    pr.to_csv(OUT_ROOT / "paired_all.tsv", sep="\t", index=False)
    sc = pd.concat([pd.read_csv(f, sep="\t", dtype={"step": str}) for f in sorted(glob.glob(str(OUT_ROOT / "*" / "*" / "scores_*.tsv")))], ignore_index=True)
    sc.to_csv(OUT_ROOT / "scores_all.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pr[["bench", "task", "step", "a", "b", "a_choosing", "b_choosing", "delta_choosing", "ci_low", "ci_high", "verdict", "delta_recall_at_5pct"]].round(3).to_string(index=False))
    print("\nwin rule per step and model: interval above zero on >= 1 cell and below zero on none")
    for (step, a), g in pr[pr.step != "3.7"].groupby(["step", "a"]):
        if step == "3.4":   # the average wins only against the better single model of each cell (protocol 3.4)
            better = g.loc[g.groupby(["bench", "task"]).b_choosing.idxmax().dropna()]
            g = pd.concat([better, g[g.b_choosing.isna() & ~g.set_index(["bench", "task"]).index.isin(better.set_index(["bench", "task"]).index)]])
        wins = int((g.ci_low > 0).sum()); losses = int((g.ci_high < 0).sum())
        print(f"  {step:4s} {a:32s} cells {len(g):2d}  wins {wins}  losses {losses}  -> {'WIN' if wins and not losses else 'no'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bench", choices=list(PATTERNS))
    ap.add_argument("--task", choices=TASKS, help="default: all four")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-plantings", type=int, help="smoke test only")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--compose", action="append", default=[], help='3.8: model:steps, e.g. "reference:3.3,3.1" or "average:3.2" (repeatable)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.pool:
        pool(); return 0
    if not a.bench:
        raise SystemExit("--bench is required")
    steps = [s for s in a.steps.split(",") if s]
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
    folds = pd.read_csv(SPLITS / "dev_folds.tsv", sep="\t").set_index("Run_accession")["fold"]
    files = sorted(glob.glob(str(F.PLANT_DIR / PATTERNS[a.bench])))[: a.max_plantings]
    plants = [pd.read_csv(f, sep="\t").set_index("Run_accession") for f in files]
    logger.info("%s benchmark: %d plantings, steps %s", a.bench, len(plants), steps)
    if a.compose:
        specs = {sp.split(":")[0]: sp.split(":")[1].split(",") for sp in a.compose}
        for task in ([a.task] if a.task else TASKS):
            run_compose(a.bench, task, specs, plants, meta, folds, a.n_boot)
        return 0
    if "3.7" in steps:
        if a.bench != "record":
            logger.info("3.7 is defined on the whole-record benchmark only; skipped")
        else:
            run_combined(a.bench, plants, meta, folds, a.n_boot)
        steps = [s for s in steps if s != "3.7"]
    if steps:
        for task in ([a.task] if a.task else TASKS):
            run_task(a.bench, task, steps, plants, meta, folds, a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
