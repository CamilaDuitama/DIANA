#!/usr/bin/env python3
"""Pick each arm's epoch budget on the pooled dev folds, then re-decide A6 on the same folds.

Input: `results/epoch_budget_v9/<arm>/fold<k>/` from
`scripts/training/04_epoch_budget_dev_folds.py`, five folds per arm.

Rules, fixed 2026-09-25 before any fit ran (also in PROJECT.md, section 4, X1):

* Criterion per epoch: `f1_macro_eligible` over the five folds' predictions pooled and
  scored ONCE, eligibility from `data/splits_v9/class_eligibility.tsv` (the table the
  hyperparameter searches used). For the multi-task net the criterion is the mean over
  its four tasks, as in the `eligible_pooled` search objective; for a single-task net it
  is its one task.
* The curve is smoothed with a centred 5-epoch moving average and the budget is the
  argmax. If the argmax lies within 5 epochs of the cap the arm has not peaked: the cap
  is raised and the arm re-run, and nothing is reported from it.
* **Amendment, 2026-09-25, after the train-set gate and before any held-out read.** The
  budget may not point past the epoch at which the pooled (mean over folds, same
  smoothing) TRAINING loss is lowest. Under the first rule `community_type` (lr 5.3e-3)
  drew 418 epochs from a one-epoch spike in a curve whose training loss had been rising
  since epoch 15, and the full-data fit had collapsed onto the majority class (train F1
  0.36, `results/final_eval_budget_v9_rule_v1_uncapped/`); `material` (lr 5.6e-3) drew
  356, three epochs after a loss spike. A model whose training loss is rising is
  diverging, and the searched learning rates were only ever used under early stopping
  at 25 to 50 epochs. The guard reads training loss only, is parameter-free, and is
  applied to every arm; it changes nothing for an arm whose loss is still falling.
  The uncapped selection is kept in `rule_v1_uncapped/` and reported alongside.
* A6 on the dev folds: for each task, the paired difference single minus multi-task,
  each at its own budget, bootstrapped 2,000 times over `archive_project`. Per task the
  verdict is single if the interval is above zero, multi if below, tie otherwise.
  DIANA is one architecture for all four tasks: if one arm wins at least one task and
  loses none, that arm; otherwise (all ties, or a split) the published single-task
  architecture stands, because nothing favours a change.
* Nothing here reads held-out.

    ./env/bin/python scripts/analysis/37_epoch_budget_select.py
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
BASE = ROOT / "results/epoch_budget_v9"

TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = {"multitask": TASKS, **{t: [t] for t in TASKS}}
N_FOLDS = 5
WINDOW = 5
N_BOOT = 2000
SEED = 42
UNLABELLED = -100


def rng_for(task: str, name: str) -> np.random.Generator:
    """Seed derived from the names, so an interval never depends on what else is scored."""
    digest = hashlib.sha256(f"{SEED}|{task}|{name}".encode()).hexdigest()[:8]
    return np.random.default_rng(int(digest, 16))


def load_arm(arm: str) -> dict:
    """Pooled per-epoch predictions of one arm over its five folds."""
    runs, fold_of, preds, classes, n_epochs = [], [], {}, None, None
    for k in range(N_FOLDS):
        d = BASE / arm / f"fold{k}"
        if not (d / "preds_by_epoch.npz").exists():
            raise FileNotFoundError(f"{arm}: fold {k} has no predictions")
        cls = json.load(open(d / "label_classes.json"))
        if classes is None:
            classes = cls
        elif cls != classes:
            raise ValueError(f"{arm}: label space differs between folds")
        r = [l.strip() for l in open(d / "eval_runs.txt") if l.strip()]
        z = np.load(d / "preds_by_epoch.npz")
        for t in ARMS[arm]:
            a = z[t]
            if n_epochs is None:
                n_epochs = a.shape[0]
            elif a.shape[0] != n_epochs:
                raise ValueError(f"{arm}: fold {k} ran {a.shape[0]} epochs, expected {n_epochs}")
            preds.setdefault(t, []).append(a)
        runs += r
        fold_of += [k] * len(r)
    runs = np.asarray(runs)
    if len(set(runs)) != len(runs):
        raise ValueError(f"{arm}: a run appears in more than one fold")
    return {"runs": runs, "fold": np.asarray(fold_of), "classes": classes,
            "pred": {t: np.concatenate(v, axis=1) for t, v in preds.items()},
            "n_epochs": n_epochs}


def truth(meta: pd.DataFrame, runs: np.ndarray, task: str, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    vals = meta.set_index("Run_accession").loc[runs, task]
    return np.asarray([UNLABELLED if pd.isna(v) else idx[str(v)] for v in vals], dtype=np.int64)


def f1_elig(y: np.ndarray, p: np.ndarray, eligible: set) -> float:
    labs = sorted(c for c in set(y.tolist()) if c in eligible)
    return f1_score(y, p, labels=labs, average="macro", zero_division=0) if labs else np.nan


def smooth(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rolling(WINDOW, center=True, min_periods=1).mean().to_numpy()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    meta = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    project = meta.set_index("Run_accession")["archive_project"]
    elig_tbl = pd.read_csv(SPLITS / "class_eligibility.tsv", sep="\t")
    elig_names = {t: set(elig_tbl[(elig_tbl.target == t) & elig_tbl.evaluable]["class"].astype(str))
                  for t in TASKS}

    arms, budgets, curves = {}, [], []
    for arm, tasks in ARMS.items():
        a = load_arm(arm)
        a["y"], a["elig"] = {}, {}
        per_task = {}
        for t in tasks:
            a["y"][t] = truth(meta, a["runs"], t, a["classes"][t])
            a["elig"][t] = {i for i, c in enumerate(a["classes"][t]) if c in elig_names[t]}
            lab = a["y"][t] != UNLABELLED
            per_task[t] = np.asarray([f1_elig(a["y"][t][lab], a["pred"][t][e][lab], a["elig"][t])
                                      for e in range(a["n_epochs"])])
        crit = np.nanmean(np.vstack([per_task[t] for t in tasks]), axis=0)
        sm = smooth(crit)
        best_uncapped = int(np.nanargmax(sm))
        # Amendment (docstring): no budget past the pooled training-loss minimum.
        loss = np.mean([json.load(open(BASE / arm / f"fold{k}" / "history.json"))["train_loss"]
                        for k in range(N_FOLDS)], axis=0)
        cap = int(np.argmin(smooth(loss))) + 1
        best = int(np.nanargmax(sm[:cap]))
        at_cap = best >= a["n_epochs"] - WINDOW
        a["budget"] = best + 1
        arms[arm] = a
        budgets.append({"arm": arm, "n_epochs_run": a["n_epochs"], "budget": best + 1,
                        "criterion_smoothed": float(sm[best]), "criterion_raw": float(crit[best]),
                        "peaked": not at_cap, "loss_cap_epoch": cap,
                        "budget_uncapped": best_uncapped + 1, "capped": best_uncapped + 1 > cap,
                        **{f"f1_elig_{t}": float(per_task[t][best]) for t in tasks}})
        curve = pd.DataFrame({"epoch": np.arange(1, a["n_epochs"] + 1), "criterion": crit,
                              "criterion_smoothed": sm, **{f"f1_elig_{t}": per_task[t] for t in tasks}})
        curve.to_csv(BASE / f"curve_{arm}.tsv", sep="\t", index=False)
        logger.info("%s: budget %d epochs (smoothed criterion %.4f; loss cap %d, uncapped argmax %d)%s",
                    arm, best + 1, sm[best], cap, best_uncapped + 1,
                    "" if not at_cap else "  NOT PEAKED: raise the cap and re-run")
        for t in tasks:
            e = best
            pd.DataFrame({"Run_accession": a["runs"], "fold": a["fold"],
                          f"{t}_pred": [a["classes"][t][i] for i in a["pred"][t][e]],
                          f"{t}_true": [None if v == UNLABELLED else a["classes"][t][v] for v in a["y"][t]]}
                         ).to_csv(BASE / f"oof_{arm}_{t}.tsv", sep="\t", index=False)
    bud = pd.DataFrame(budgets)
    bud.to_csv(BASE / "epoch_budget.tsv", sep="\t", index=False)

    rows = []
    for t in TASKS:
        s, m = arms[t], arms["multitask"]
        if not np.array_equal(s["runs"], m["runs"]):
            raise ValueError(f"{t}: single and multi-task arms scored different runs")
        y = s["y"][t]
        if not np.array_equal(y, m["y"][t]):
            raise ValueError(f"{t}: label spaces differ between arms")
        lab = y != UNLABELLED
        ps, pm = s["pred"][t][s["budget"] - 1][lab], m["pred"][t][m["budget"] - 1][lab]
        yy, g = y[lab], project.loc[s["runs"][lab]].to_numpy()
        elig = s["elig"][t]
        uniq = np.unique(g)
        by = {p: np.where(g == p)[0] for p in uniq}
        obs = f1_elig(yy, ps, elig) - f1_elig(yy, pm, elig)
        rng = rng_for(t, "a6:single-multi")
        dd = []
        for _ in range(N_BOOT):
            sel = np.concatenate([by[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            dd.append(f1_elig(yy[sel], ps[sel], elig) - f1_elig(yy[sel], pm[sel], elig))
        dd = np.asarray(dd, float)
        dd = dd[np.isfinite(dd)]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        rows.append({"task": t, "n_runs": int(lab.sum()), "n_projects": len(uniq),
                     "budget_single": s["budget"], "budget_multi": m["budget"],
                     "f1_single": f1_elig(yy, ps, elig), "f1_multi": f1_elig(yy, pm, elig),
                     "delta_single_minus_multi": obs, "ci_low": lo, "ci_high": hi,
                     "frac_boot_positive": float((dd > 0).mean()),
                     "verdict": "single" if lo > 0 else ("multi" if hi < 0 else "tie")})
    a6 = pd.DataFrame(rows)
    a6.to_csv(BASE / "a6_devfold_paired.tsv", sep="\t", index=False)

    wins = a6.verdict.value_counts().to_dict()
    n_single, n_multi = wins.get("single", 0), wins.get("multi", 0)
    if n_single and not n_multi:
        decision = "single-task"
    elif n_multi and not n_single:
        decision = "multi-task"
    else:
        decision = "single-task (published architecture stands: no arm wins without losing)"
    if not bud.peaked.all():
        decision = "UNDECIDED: " + ", ".join(bud[~bud.peaked].arm) + " did not peak; raise the cap"
    (BASE / "decision.txt").write_text(
        f"A6 on the dev folds, rule fixed 2026-09-25: {decision}\n"
        f"per-task verdicts: {wins}\n")

    pd.set_option("display.width", 200)
    print("\nepoch budgets, pooled dev-fold f1_macro_eligible, 5-epoch centred smoothing:")
    print(bud.round(4).to_string(index=False))
    print("\nA6 on the dev folds, paired over BioProjects (single minus multi-task):")
    print(a6.round(4).to_string(index=False))
    print(f"\ndecision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
