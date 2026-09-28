#!/usr/bin/env python
"""Fixed-epoch fit of one arm on four grouped dev folds, scored on the fifth at every epoch.

Why this exists (X1, 2026-09-25)
--------------------------------
Early stopping needs a validation criterion, and once the inner split is grouped by
BioProject (as it must be) a single-task net has no usable one: `feature` had 19
labelled validation runs and `sample_host` 4 distinct classes, so the criterion could
not rank epochs and both arms were left underfit (train F1 0.14 / 0.17,
`results/final_fixed_v9/`). The multi-task net averages four heads and was fine.

So the number of epochs is chosen the way every other hyperparameter already is: on
the 5 BioProject-grouped dev folds. This script trains one arm on four folds for a
fixed number of epochs with NO inner split and records the argmax prediction for every
run of the fifth fold after every epoch. `scripts/analysis/37_epoch_budget_select.py`
pools the five folds, scores each epoch once over all training projects, and picks the
budget. The final fit then trains on all 2,716 runs for that many epochs.

Fairness: the same rule, cap and criterion are applied to the multi-task net and to
each single-task net. Hyperparameters, features, ids, seed, tau and label smoothing are
those of `configs/final_fixed_v9/` (the searched values), unchanged. Baselines have no
early stopping and are untouched. Held-out is not read here.

Outputs, per arm and fold, under --out:
  preds_by_epoch.npz   one int16 array per task, shape [n_epochs, n_eval]; -1 = unlabelled
  eval_runs.txt        Run_accession of the eval fold, in array column order
  label_classes.json   class list per task (LabelEncoder fitted on all training labels,
                       so indices match the final model's label space)
  history.json         train loss and eval-fold macro-F1 per epoch
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from diana.data.loader import MatrixLoader
from diana.data.validation_split import GROUP_COL
from diana.models.multitask_mlp import IGNORE_INDEX, MultiTaskMLP
from diana.training.trainer import MultiTaskTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_FOLDS = 5
PREDICT_BATCH = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", type=Path, help="a configs/final_fixed_v9/<arm>.json")
    p.add_argument("--fold", type=int, required=True, help="dev fold held out for scoring (0-4)")
    p.add_argument("--out", type=Path, required=True, help="output directory for this fold")
    p.add_argument("--max-epochs", type=int, default=600,
                   help="fixed cap, identical for every arm; the selector checks the "
                        "argmax is not at the cap")
    p.add_argument("--dev-folds", type=Path, default=Path("data/splits_v9/dev_folds.tsv"),
                   help="the 5 BioProject-grouped dev folds of train")
    return p.parse_args()


def load_train_matrix(config: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """The 2,716 training runs, in matrix order, refusing any mismatch with the split."""
    loader = MatrixLoader(Path(config["features_path"]))
    X_all, meta_pl = loader.load_with_metadata(
        metadata_path=Path(config["metadata_path"]),
        align_to_matrix=True,
        require_all_metadata=True,
    )
    meta = meta_pl.to_pandas()
    train_ids = {l.strip() for l in open(config["train_ids_path"]) if l.strip()}
    mask = meta["Run_accession"].isin(train_ids).to_numpy()
    X, meta = X_all[mask], meta[mask].reset_index(drop=True)
    if len(meta) != len(train_ids):
        raise ValueError(f"{len(train_ids)} train ids but {len(meta)} found in the matrix")
    return X, meta


def encode_labels(meta: pd.DataFrame, task_names: list[str]) -> tuple[dict, dict]:
    """Masked LabelEncoder per task, fitted on ALL training labels (as the final fit does)."""
    y, classes = {}, {}
    for task in task_names:
        vals = meta[task].values
        present = pd.notna(vals)
        enc = LabelEncoder().fit(vals[present])
        arr = np.full(len(vals), IGNORE_INDEX, dtype=np.int64)
        arr[present] = enc.transform(vals[present])
        y[task] = arr
        classes[task] = [str(c) for c in enc.classes_]
    return y, classes


def fold_assignment(meta: pd.DataFrame, dev_folds: Path) -> np.ndarray:
    dev = pd.read_csv(dev_folds, sep="\t")
    if set(dev.Run_accession) != set(meta.Run_accession):
        raise ValueError("dev_folds.tsv and the training matrix hold different runs")
    return meta.Run_accession.map(dev.set_index("Run_accession")["fold"]).to_numpy()


def predict_argmax(model: MultiTaskMLP, X_eval: torch.Tensor, task_names: list[str]) -> dict:
    model.eval()
    out = {t: [] for t in task_names}
    with torch.no_grad():
        for i in range(0, len(X_eval), PREDICT_BATCH):
            logits = model(X_eval[i:i + PREDICT_BATCH])
            for t in task_names:
                out[t].append(logits[t].argmax(dim=1).cpu().numpy().astype(np.int16))
    return {t: np.concatenate(v) for t, v in out.items()}


def main() -> int:
    args = parse_args()
    config = json.load(open(args.config))
    for key in ("sequence_encoder", "sequence_channel", "attention_pool"):
        if config.get(key):
            raise ValueError(f"{key} arms are out of scope for the epoch budget")
    if not 0 <= args.fold < N_FOLDS:
        raise ValueError(f"--fold must be in 0..{N_FOLDS - 1}")
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(args.out / "arm_config.json", "w"), indent=2)

    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    task_names = list(config["task_names"])
    hp = config["hyperparameters"]
    X, meta = load_train_matrix(config)
    y, classes = encode_labels(meta, task_names)
    folds = fold_assignment(meta, args.dev_folds)

    eval_mask = folds == args.fold
    train_idx, eval_idx = np.where(~eval_mask)[0], np.where(eval_mask)[0]
    groups = meta[GROUP_COL].to_numpy()
    shared = set(groups[train_idx]) & set(groups[eval_idx])
    if shared:
        raise ValueError(f"{len(shared)} BioProject(s) on both sides of fold {args.fold}: "
                         f"{sorted(shared)[:5]}")
    logger.info("fold %d: %d train runs / %d eval runs, %d / %d projects, 0 shared",
                args.fold, len(train_idx), len(eval_idx),
                len(set(groups[train_idx])), len(set(groups[eval_idx])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)
    n_classes = {t: len(classes[t]) for t in task_names}
    model = MultiTaskMLP(input_dim=X.shape[1], num_classes=n_classes, **hp["model_params"])

    # Priors from the rows the model is fitted on, never from the eval fold: a class with
    # no training rows in this fold keeps a zero count (see compute_class_priors in
    # 01_train_multitask_single_fold.py for what a finite prior does to such a class).
    tau = float((config.get("class_imbalance") or {}).get("logit_adjust_tau", 0.0) or 0.0)
    priors = None
    if tau > 0:
        priors = {}
        for t in task_names:
            lab = y[t][train_idx]
            lab = lab[lab != IGNORE_INDEX]
            priors[t] = torch.from_numpy(np.bincount(lab, minlength=n_classes[t]).astype(np.float32))
    label_smoothing = config.get("label_smoothing_per_task", config.get("label_smoothing", 0.0))
    trainer = MultiTaskTrainer(
        model=model, task_names=task_names, device=device,
        learning_rate=hp["trainer_params"]["learning_rate"],
        weight_decay=hp["trainer_params"]["weight_decay"],
        task_weights=hp["trainer_params"]["task_weights"],
        class_weights=None, class_priors=priors, logit_adjust_tau=tau,
        label_smoothing=label_smoothing,
    )

    gen = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(torch.FloatTensor(X[train_idx]),
                             *[torch.LongTensor(y[t][train_idx]) for t in task_names])
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, generator=gen)
    X_eval = torch.FloatTensor(X[eval_idx]).to(device)
    y_eval = {t: y[t][eval_idx] for t in task_names}

    preds = {t: np.full((args.max_epochs, len(eval_idx)), -1, dtype=np.int16) for t in task_names}
    history = {"train_loss": [], "eval_macro_f1": {t: [] for t in task_names}}
    for epoch in range(args.max_epochs):
        tr = trainer.train_one_epoch(train_loader)
        history["train_loss"].append(float(tr["loss"]))
        ep = predict_argmax(model, X_eval, task_names)
        for t in task_names:
            preds[t][epoch] = ep[t]
            lab = y_eval[t] != IGNORE_INDEX
            f1 = (f1_score(y_eval[t][lab], ep[t][lab], average="macro", zero_division=0)
                  if lab.any() else float("nan"))
            history["eval_macro_f1"][t].append(float(f1))
        if (epoch + 1) % 25 == 0:
            logger.info("epoch %d/%d  train loss %.4f  eval macro-F1 %s", epoch + 1,
                        args.max_epochs, tr["loss"],
                        {t: round(history["eval_macro_f1"][t][-1], 3) for t in task_names})

    np.savez_compressed(args.out / "preds_by_epoch.npz", **preds)
    (args.out / "eval_runs.txt").write_text("\n".join(meta.Run_accession.to_numpy()[eval_idx]) + "\n")
    json.dump(classes, open(args.out / "label_classes.json", "w"), indent=2)
    json.dump(history, open(args.out / "history.json", "w"))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
