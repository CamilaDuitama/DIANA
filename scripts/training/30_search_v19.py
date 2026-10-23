#!/usr/bin/env python3
"""v19 hyperparameter search for one arm (PROTOCOLS.md v19, C3).

Each trial trains every dev fold to a fixed cap of epochs, with the classes absent
from the fold's fit rows masked (C1), and records the out-of-fold probabilities after
every epoch. The trial's score is the maximum of the pooled out-of-fold criterion
curve: scored runs only, macro-F1 over the eligible classes, five folds pooled (C2),
the mean over the four tasks for the multi-task arm; smoothed over 5 epochs and capped
at the minimum of the mean training loss. The trial's epoch count is that argmax. No
checkpoint is ever chosen on a fold being graded. 100 TPE trials (sampler seed 42) over
the v9 search space; every fold fit uses seed 42, so trials differ only by their
hyperparameters. The study is stored in SQLite, so an interrupted search resumes.

`evaluate_config` is shared with 31_rescore_top5_v19.py, which re-scores the five best
trials with five seeds (C4): the seeds' probabilities are averaged per epoch before the
argmax, as the deployed five-seed ensemble does.

    ./env/bin/python scripts/training/30_search_v19.py --arm feature \\
        --split-dir data/splits_v19 --matrix data/matrices/matrix_v17_train/unitigs.frac.mat \\
        --out results/search_v19/feature
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from diana.evaluation import scoring as S

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["community_type", "feature", "sample_host", "material"]
ARMS = {**{t: [t] for t in TASKS}, "multitask": TASKS}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("search_v19")


def fold_module():
    """The dev-fold fitter (04_epoch_budget_dev_folds.py), imported once."""
    if "fold_fit" in sys.modules:
        return sys.modules["fold_fit"]
    spec = importlib.util.spec_from_file_location("fold_fit", ROOT / "scripts/training/04_epoch_budget_dev_folds.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fold_fit"] = mod
    spec.loader.exec_module(mod)
    return mod


def base_config(arm: str, split_dir: Path, matrix: Path) -> dict:
    tasks = ARMS[arm]
    return {"features_path": str(matrix),
            "metadata_path": str(split_dir / "train_metadata.tsv"),
            "train_ids_path": str(split_dir / "train_accessions.txt"),
            "task_names": tasks, "task_types": {t: "classification" for t in tasks},
            "random_seed": 42}


def suggest(trial: optuna.Trial, tasks: list[str]) -> dict:
    """The v9 search space (manuscript Supplementary Table, search space)."""
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden = [trial.suggest_int(f"hidden_dim_{i}", 64, 512, step=64) for i in range(n_layers)]
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    bn = trial.suggest_categorical("use_batch_norm", [True, False])
    act = trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"])
    tau = trial.suggest_float("logit_adjust_tau", 0.05, 0.8)
    weights = ({t: trial.suggest_float(f"task_weight_{t}", 0.5, 2.0) for t in tasks}
               if len(tasks) > 1 else {tasks[0]: 1.0})
    ls = {t: trial.suggest_float(f"ls_{t}", 0.0, 0.15) for t in tasks}
    return hp_config(hidden, dropout, act, bn, lr, wd, weights, batch, tau, ls)


def hp_config(hidden, dropout, act, bn, lr, wd, weights, batch, tau, ls) -> dict:
    return {"hyperparameters": {"model_params": {"hidden_dims": list(hidden), "dropout": float(dropout),
                                                 "activation": act, "use_batch_norm": bool(bn)},
                                "trainer_params": {"learning_rate": float(lr), "weight_decay": float(wd),
                                                   "task_weights": dict(weights)},
                                "batch_size": int(batch)},
            "class_imbalance": {"method": "logit_adjustment", "logit_adjust_tau": float(tau)},
            "label_smoothing_per_task": dict(ls)}


def params_to_hp(params: dict, tasks: list[str]) -> dict:
    """Rebuild the hp part of a config from a trial's stored parameters."""
    hidden = [params[f"hidden_dim_{i}"] for i in range(params["n_layers"])]
    weights = ({t: params[f"task_weight_{t}"] for t in tasks} if len(tasks) > 1 else {tasks[0]: 1.0})
    return hp_config(hidden, params["dropout"], params["activation"], params["use_batch_norm"],
                     params["learning_rate"], params["weight_decay"], weights, params["batch_size"],
                     params["logit_adjust_tau"], {t: params[f"ls_{t}"] for t in tasks})


def evaluate_config(config: dict, data: dict, cap: int, seeds: list[int], eligible: dict,
                    scored: dict) -> dict:
    """Fit every dev fold for `cap` epochs per seed; pooled, seed-averaged criterion curve."""
    fm = fold_module()
    tasks = data["task_names"]
    n = len(data["meta"])
    pos = pd.Series(np.arange(n), index=data["meta"].Run_accession.to_numpy())
    preds = {t: np.full((cap, n), -1, dtype=np.int16) for t in tasks}
    losses = []
    for k in range(5):
        psum, eval_runs = None, None
        for s in seeds:
            args = fm.parse_args(["unused.json", "--fold", str(k), "--out", "unused", "--dev-folds", "unused",
                                  "--max-epochs", str(cap), "--seed", str(s), "--save-probs-by-epoch"])
            res = fm.fit_fold(args, config, data)
            if res["n_epochs_run"] != cap:
                raise RuntimeError(f"fold {k} seed {s} ran {res['n_epochs_run']} of {cap} epochs")
            if eval_runs is None:
                eval_runs = list(res["eval_runs"])
            elif list(res["eval_runs"]) != eval_runs:
                raise RuntimeError(f"fold {k}: seeds scored different runs")
            pr = {t: res["probs_by_epoch"][t].astype(np.float32) for t in tasks}
            psum = pr if psum is None else {t: psum[t] + pr[t] for t in tasks}
            losses.append(res["history"]["train_loss"])
        cols = pos.loc[eval_runs].to_numpy()
        for t in tasks:
            preds[t][:, cols] = psum[t].argmax(axis=2)
    if any((preds[t] < 0).any() for t in tasks):
        raise RuntimeError("some training run has no out-of-fold prediction")
    classes = data["classes"]
    curves = {}
    for t in tasks:
        ok = scored[t]
        y = data["meta"][t].astype(str).to_numpy()[ok]
        names = np.asarray(classes[t])
        curves[t] = np.array([S.f1_macro_eligible(y, names[preds[t][e, ok]], eligible[t]) for e in range(cap)])
    crit = np.nanmean(np.vstack([curves[t] for t in tasks]), axis=0)
    loss = np.mean(np.asarray(losses, float), axis=0)
    sel = S.select_epoch(crit, loss)
    return {"selection": sel, "criterion": crit, "loss": loss, "curves": curves,
            "f1_at_epoch": {t: float(curves[t][sel["epochs"] - 1]) for t in tasks}}


def prepare(arm: str, split_dir: Path, matrix: Path) -> tuple[dict, dict, dict, dict]:
    """Base config, loaded data, eligible classes and scored-run masks for an arm."""
    fm = fold_module()
    config = base_config(arm, split_dir, matrix)
    data = fm.load_data(config, split_dir / "dev_folds.tsv")
    S.check_same_split(data["meta"], open(split_dir / "train_accessions.txt").read().split(), "training matrix")
    tasks = ARMS[arm]
    eligible = {t: S.eligible_classes(split_dir / "class_eligibility.tsv", t) for t in tasks}
    folds = pd.Series(data["folds"], index=data["meta"].Run_accession.to_numpy())
    scored = {}
    for t in tasks:
        labels = pd.Series(data["meta"][t].replace("", np.nan).to_numpy(), index=folds.index)
        scored[t] = S.scored_devfold(labels, folds).to_numpy()
        log.info("%s: %d labelled runs, %d scored on the dev folds; %d eligible classes", t,
                 int(labels.notna().sum()), int(scored[t].sum()), len(eligible[t]))
    return config, data, eligible, scored


def write_curve(path: Path, res: dict) -> None:
    pd.DataFrame({"epoch": np.arange(1, len(res["criterion"]) + 1), "criterion": res["criterion"],
                  "criterion_smoothed": S.smooth(res["criterion"]), "train_loss": res["loss"],
                  **{f"f1_{t}": v for t, v in res["curves"].items()}}).to_csv(path, sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--sampler-seed", type=int, default=42)
    a = ap.parse_args()
    (a.out / "curves").mkdir(parents=True, exist_ok=True)
    base, data, eligible, scored = prepare(a.arm, a.split_dir, a.matrix)
    tasks = ARMS[a.arm]

    def objective(trial: optuna.Trial) -> float:
        config = {**base, **suggest(trial, tasks)}
        res = evaluate_config(config, data, a.cap, [42], eligible, scored)
        sel = res["selection"]
        value = sel["criterion_smoothed"]
        if not np.isfinite(value):
            raise ValueError(f"trial {trial.number}: non-finite score")
        for k, v in sel.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("f1_at_epoch", res["f1_at_epoch"])
        write_curve(a.out / "curves" / f"trial_{trial.number:03d}.tsv", res)
        log.info("trial %d: %.4f at %d epochs (peaked %s, loss cap %d)", trial.number, value,
                 sel["epochs"], sel["peaked"], sel["loss_cap_epoch"])
        return value

    study = optuna.create_study(direction="maximize", study_name=f"v19_{a.arm}",
                                sampler=optuna.samplers.TPESampler(seed=a.sampler_seed),
                                storage=f"sqlite:///{a.out / 'study.db'}", load_if_exists=True)
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if done < a.n_trials:
        study.optimize(objective, n_trials=a.n_trials - done)
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials) < a.n_trials:
        raise RuntimeError(f"{len(trials)} of {a.n_trials} trials completed")
    table = pd.DataFrame([{"trial": t.number, "score": t.value, **t.params,
                           **{k: v for k, v in t.user_attrs.items() if k != "f1_at_epoch"}} for t in trials])
    table.sort_values("score", ascending=False).to_csv(a.out / "trials.tsv", sep="\t", index=False)
    top = sorted(trials, key=lambda t: t.value, reverse=True)[:5]
    json.dump([{"rank": i, "trial": t.number, "score": t.value, "params": t.params,
                "epochs": t.user_attrs["epochs"], "cap": a.cap,
                "config": {**base, **params_to_hp(t.params, tasks)}} for i, t in enumerate(top)],
              open(a.out / "top5.json", "w"), indent=1)
    log.info("top 5: %s", [(t.number, round(t.value, 4)) for t in top])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
