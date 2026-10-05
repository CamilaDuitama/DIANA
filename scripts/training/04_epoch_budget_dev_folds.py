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
import copy
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
from diana.data.validation_split import GROUP_COL, grouped_validation_split
from diana.models.multitask_mlp import IGNORE_INDEX, MultiTaskMLP
from diana.models.residual_linear import ResidualOnLinear
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
    p.add_argument("--allow-shared-projects", action="store_true",
                   help="W protocol only: folds grouped by sample, not by BioProject, so studies sit on "
                        "both sides by design; the disjointness assertion is skipped and logged")
    p.add_argument("--seed", type=int, default=None, help="override the config's random_seed (X7b, seeds)")
    p.add_argument("--lr", type=float, default=None, help="override the learning rate (X6)")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="X6: cosine annealing of the learning rate to zero over --max-epochs, stepped per epoch")
    p.add_argument("--lr-horizon", type=int, default=None,
                   help="T_max of the cosine schedule when it differs from --max-epochs (the reference material arm: "
                        "cosine over 300 epochs, run for its 66-epoch budget)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="X7a: per-epoch exponential moving average of the weights with this decay, started "
                        "after the first epoch; predictions come from the averaged weights (0 = off)")
    p.add_argument("--early-stop", choices=["none", "random", "grouped"], default="none",
                   help="X5: hold --val-split of the training-fold runs out of the fit (random over runs, or "
                        "grouped by BioProject with the class floor), stop after --patience epochs without "
                        "improvement of the inner macro-F1, and record the selected epoch")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--offsets-dir", type=Path, default=None,
                   help="X9a: directory with offsets_<task>_fold<k>.npz (18_logreg_offsets_v9.py); the fixed "
                        "logistic-regression logits are added to the network's logits in training and at prediction")
    p.add_argument("--linear-init-dir", type=Path, default=None,
                   help="Phase 4 step 4.1: directory with linear_<task>_fold<k>.npz (19_logreg_init_v9.py); the model becomes "
                        "logistic regression + a zero-initialised correction network, the linear part at L2 = 1/(C n_fit)")
    p.add_argument("--study-weights", action="store_true",
                   help="Phase 4 step 4.2: loss weight 1/max(n_study, 5) per fit row of the task, normalised to mean 1")
    p.add_argument("--depth-augment", type=float, default=0.0,
                   help="Phase 4 step 4.3: probability per epoch of thinning a fit row within its class's depth range")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Phase 4 step 4.5: Beta(alpha, alpha) mixup of same-class rows from different studies, half of each batch")
    p.add_argument("--adversarial", type=float, default=0.0,
                   help="Phase 4 step 4.4: gradient-reversal scale of a head predicting the study from the backbone features")
    p.add_argument("--heldout", type=Path, default=None,
                   help="FINAL TEST ONLY (ledger read, Camila's approval): fit on all training runs and predict this held-out "
                        "matrix (unitigs.frac.mat) for the runs in --heldout-ids with labels from --heldout-metadata; --fold is ignored")
    p.add_argument("--heldout-ids", type=Path, default=Path("data/splits_v9/test_accessions.txt"))
    p.add_argument("--heldout-metadata", type=Path, default=Path("data/splits_v9/test_metadata.tsv"))
    p.add_argument("--resubstitution", action="store_true",
                   help="SELF-GRADING DEMO ONLY (PROTOCOLS.md 2026-10-05): train on all training runs and predict those same runs; --fold is ignored")
    p.add_argument("--adversarial-mode", choices=["reverse", "confusion"], default="reverse",
                   help="4.4: 'reverse' = gradient reversal (unbounded); 4.4b: 'confusion' = uniform-target cross-entropy, bounded")
    p.add_argument("--save-probs", action="store_true",
                   help="also save softmax probabilities of the last epoch run (X7b, seed ensembles)")
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


def predict_argmax(model: MultiTaskMLP, X_eval: torch.Tensor, task_names: list[str], offsets: dict | None = None) -> dict:
    model.eval()
    out = {t: [] for t in task_names}
    with torch.no_grad():
        for i in range(0, len(X_eval), PREDICT_BATCH):
            logits = model(X_eval[i:i + PREDICT_BATCH])
            for t in task_names:
                lg = logits[t] + offsets[t][i:i + PREDICT_BATCH] if offsets is not None else logits[t]
                out[t].append(lg.argmax(dim=1).cpu().numpy().astype(np.int16))
    return {t: np.concatenate(v) for t, v in out.items()}


def load_offsets(offsets_dir: Path, fold: int, task_names: list[str], runs: np.ndarray, classes: dict) -> dict:
    """Per-task logit offsets aligned to the matrix rows; class order must match the encoder."""
    out = {}
    for t in task_names:
        z = np.load(offsets_dir / f"offsets_{t}_fold{fold}.npz", allow_pickle=False)
        if list(z["classes"].astype(str)) != list(classes[t]):
            raise ValueError(f"{t}: offset class order differs from the label encoder")
        pos = pd.Series(np.arange(len(z["runs"])), index=z["runs"].astype(str)).reindex(runs)
        if pos.isna().any():
            raise ValueError(f"{t}: {int(pos.isna().sum())} runs without an offset")
        out[t] = z["offsets"][pos.to_numpy().astype(int)].astype(np.float32)
    return out


def load_linear_init(init_dir: Path, fold: int, task_names: list[str], classes: dict, n_features: int) -> tuple[dict, dict]:
    """4.1: logistic-regression weights per task in the encoder's class order (absent classes zero), and L2 = 1/(C n_fit)."""
    init, l2 = {}, {}
    for t in task_names:
        z = np.load(init_dir / f"linear_{t}_fold{fold}.npz", allow_pickle=False)
        coef, intercept, cls = z["coef"].astype(np.float32), z["intercept"].astype(np.float32), list(z["classes"].astype(str))
        if coef.shape[0] == 1 and len(cls) == 2:          # sklearn's binary layout: one row, P(cls[1]) = sigmoid(d)
            coef = np.vstack([-coef / 2, coef / 2]); intercept = np.array([-intercept[0] / 2, intercept[0] / 2], dtype=np.float32)
        W = np.zeros((len(classes[t]), n_features), dtype=np.float32); b = np.zeros(len(classes[t]), dtype=np.float32)
        for j, c in enumerate(cls):
            W[classes[t].index(c)] = coef[j]; b[classes[t].index(c)] = intercept[j]
        init[t] = (W, b); l2[t] = 1.0 / (float(z["C"]) * float(z["n_fit"]))
    return init, l2


def study_weights(groups: np.ndarray, labels: np.ndarray, floor: int = 5) -> np.ndarray:
    """4.2: 1/max(n_study, floor) for labelled rows (n_study = labelled fit rows of the study), mean 1; 0 for unlabelled rows."""
    lab = labels != IGNORE_INDEX
    counts = pd.Series(groups[lab]).value_counts()
    n = counts.reindex(groups).fillna(0).to_numpy(dtype=float)
    w = np.zeros(len(groups), dtype=np.float32)
    w[lab] = 1.0 / np.maximum(n[lab], floor)
    w[lab] *= lab.sum() / w[lab].sum()
    return w


def depth_floor(X_fit: np.ndarray, labels: np.ndarray, pct: float = 5.0) -> np.ndarray:
    """4.3: per fit row, the 5th percentile of the non-zero unitig count among the fit rows of its class (global for unlabelled rows)."""
    d = (X_fit > 0).sum(axis=1)
    lo = np.full(len(d), np.percentile(d, pct), dtype=np.float32)
    for c in np.unique(labels[labels != IGNORE_INDEX]):
        m = labels == c
        lo[m] = np.percentile(d[m], pct)
    return lo


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

    seed = int(args.seed if args.seed is not None else config.get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    task_names = list(config["task_names"])
    hp = config["hyperparameters"]
    X, meta = load_train_matrix(config)
    y, classes = encode_labels(meta, task_names)
    folds = fold_assignment(meta, args.dev_folds)

    heldout = None
    if args.resubstitution:
        eval_mask = np.zeros(len(meta), dtype=bool)
        logger.info("RESUBSTITUTION (self-grading demo): fitting on all %d runs and predicting the same runs", len(meta))
    elif args.heldout is not None:
        # final test: every training run fits, the evaluation rows come from the held-out matrix
        loader = MatrixLoader(args.heldout)
        Xh_all, meta_h = loader.load_with_metadata(metadata_path=args.heldout_metadata, align_to_matrix=True, require_all_metadata=True)
        meta_h = meta_h.to_pandas()
        ho_ids = {l.strip() for l in open(args.heldout_ids) if l.strip()}
        hmask = meta_h["Run_accession"].isin(ho_ids).to_numpy()
        Xh, meta_h = Xh_all[hmask], meta_h[hmask].reset_index(drop=True)
        if len(meta_h) != len(ho_ids):
            raise ValueError(f"{len(ho_ids)} held-out ids but {len(meta_h)} found in {args.heldout}")
        if set(meta_h.Run_accession) & set(meta.Run_accession):
            raise ValueError("held-out runs overlap the training runs")
        if set(meta_h[GROUP_COL]) & set(meta[GROUP_COL]):
            raise ValueError("held-out BioProjects overlap the training BioProjects")
        heldout = (Xh, meta_h)
        logger.info("FINAL TEST: fitting on all %d training runs, predicting %d held-out runs from %s", len(meta), len(meta_h), args.heldout)
        eval_mask = np.zeros(len(meta), dtype=bool)
    else:
        eval_mask = folds == args.fold
    train_idx, eval_idx = np.where(~eval_mask)[0], np.where(eval_mask)[0]
    if args.resubstitution:
        eval_idx = train_idx
    groups = meta[GROUP_COL].to_numpy()
    shared = set(groups[train_idx]) & set(groups[eval_idx]) if (heldout is None and not args.resubstitution) else set()
    if shared and not args.allow_shared_projects:
        raise ValueError(f"{len(shared)} BioProject(s) on both sides of fold {args.fold}: "
                         f"{sorted(shared)[:5]}")
    if shared:
        logger.warning("WITHIN-STUDY folds (--allow-shared-projects): %d BioProjects on both sides of fold %d; "
                       "this is not an out-of-project evaluation", len(shared), args.fold)
    logger.info("fold %d: %d train runs / %d eval runs, %d / %d projects, 0 shared",
                args.fold, len(train_idx), len(eval_idx),
                len(set(groups[train_idx])), len(set(groups[eval_idx])))

    # X5: an inner validation split inside the training folds, for early stopping only.
    fit_idx, val_idx = train_idx, None
    if args.early_stop == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx)
        n_val = max(1, int(round(args.val_split * len(train_idx))))
        val_idx, fit_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        logger.info("early stopping on a RANDOM inner split: %d fit / %d validation runs, %d projects on both sides",
                    len(fit_idx), len(val_idx), len(set(groups[fit_idx]) & set(groups[val_idx])))
    elif args.early_stop == "grouped":
        fit_idx, val_idx = grouped_validation_split(
            train_idx, groups[train_idx], validation_split=args.val_split, random_state=seed,
            task_labels={t: y[t][train_idx] for t in task_names}, ignore_index=IGNORE_INDEX)
        fit_idx, val_idx = np.sort(np.asarray(fit_idx)), np.sort(np.asarray(val_idx))
        assert not (set(groups[fit_idx]) & set(groups[val_idx]))
        logger.info("early stopping on a GROUPED inner split: %d fit / %d validation runs, %d / %d projects, 0 shared",
                    len(fit_idx), len(val_idx), len(set(groups[fit_idx])), len(set(groups[val_idx])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)
    n_classes = {t: len(classes[t]) for t in task_names}
    model = MultiTaskMLP(input_dim=X.shape[1], num_classes=n_classes, **hp["model_params"])
    if args.linear_init_dir is not None:
        lin_init, lin_l2 = load_linear_init(args.linear_init_dir, args.fold, task_names, classes, X.shape[1])
        model = ResidualOnLinear(model, lin_init, lin_l2)
        logger.info("4.1: linear part initialised from %s, L2 %s", args.linear_init_dir, {t: f"{v:.3g}" for t, v in lin_l2.items()})

    # Priors from the rows the model is fitted on, never from the eval fold: a class with
    # no training rows in this fold keeps a zero count (see compute_class_priors in
    # 01_train_multitask_single_fold.py for what a finite prior does to such a class).
    tau = float((config.get("class_imbalance") or {}).get("logit_adjust_tau", 0.0) or 0.0)
    priors = None
    if tau > 0:
        priors = {}
        for t in task_names:
            lab = y[t][fit_idx]
            lab = lab[lab != IGNORE_INDEX]
            priors[t] = torch.from_numpy(np.bincount(lab, minlength=n_classes[t]).astype(np.float32))
    label_smoothing = config.get("label_smoothing_per_task", config.get("label_smoothing", 0.0))
    n_studies_fit = len(set(groups[fit_idx]))
    trainer = MultiTaskTrainer(
        model=model, task_names=task_names, device=device,
        learning_rate=args.lr if args.lr is not None else hp["trainer_params"]["learning_rate"],
        weight_decay=hp["trainer_params"]["weight_decay"],
        task_weights=hp["trainer_params"]["task_weights"],
        class_weights=None, class_priors=priors, logit_adjust_tau=tau,
        label_smoothing=label_smoothing,
        sample_weighted=args.study_weights, depth_augment=args.depth_augment,
        uses_offsets=args.offsets_dir is not None, mixup=args.mixup,
        adversarial=args.adversarial, n_studies=n_studies_fit if args.adversarial > 0 else 0,
        adv_in_dim=hp["model_params"]["hidden_dims"][-1] if args.adversarial > 0 else 0,
        adversarial_mode=args.adversarial_mode,
    )

    offsets = None
    if args.offsets_dir is not None:
        offsets = load_offsets(args.offsets_dir, args.fold, task_names, meta.Run_accession.to_numpy(), classes)
        logger.info("X9a: logit offsets loaded from %s for %d runs", args.offsets_dir, len(meta))
    extras = []
    if args.study_weights:
        if len(task_names) != 1:
            raise ValueError("--study-weights is defined for single-task arms")
        w = study_weights(groups[fit_idx], y[task_names[0]][fit_idx])
        extras.append(torch.FloatTensor(w))
        logger.info("4.2: study weights over %d studies, min %.3f max %.3f (mean 1 over labelled rows)",
                    len(set(groups[fit_idx][y[task_names[0]][fit_idx] != IGNORE_INDEX])), w[w > 0].min(), w.max())
    if args.depth_augment > 0:
        if len(task_names) != 1:
            raise ValueError("--depth-augment is defined for single-task arms")
        lo = depth_floor(X[fit_idx], y[task_names[0]][fit_idx])
        extras.append(torch.FloatTensor(lo))
        logger.info("4.3: depth augmentation p=%.2f, class floors %d..%d non-zero unitigs", args.depth_augment, int(lo.min()), int(lo.max()))
    n_studies = 0
    if args.mixup > 0 or args.adversarial > 0:
        if len(task_names) != 1:
            raise ValueError("--mixup and --adversarial are defined for single-task arms")
        codes = pd.factorize(pd.Series(groups[fit_idx]))[0]
        n_studies = len(set(codes))
        extras.append(torch.LongTensor(codes))
        if args.mixup > 0:
            logger.info("4.5: mixup alpha=%.2f on half of each batch, same-class partners from other studies (%d studies)", args.mixup, n_studies)
        if args.adversarial > 0:
            logger.info("4.4: study-adversarial head over %d studies, strength %.2f, mode %s", n_studies, args.adversarial, args.adversarial_mode)
    gen = torch.Generator().manual_seed(seed)
    train_ds = TensorDataset(torch.FloatTensor(X[fit_idx]),
                             *[torch.LongTensor(y[t][fit_idx]) for t in task_names],
                             *([torch.FloatTensor(offsets[t][fit_idx]) for t in task_names] if offsets is not None else []),
                             *extras)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, generator=gen)
    if heldout is not None:
        Xh, meta_h = heldout
        enc_h = {t: np.array([classes[t].index(v) if isinstance(v, str) and v in classes[t] else IGNORE_INDEX
                              for v in meta_h[t].astype(object)]) for t in task_names}
        X_eval = torch.FloatTensor(Xh).to(device); y_eval = enc_h; eval_runs = meta_h.Run_accession.to_numpy()
    else:
        X_eval = torch.FloatTensor(X[eval_idx]).to(device)
        y_eval = {t: y[t][eval_idx] for t in task_names}; eval_runs = meta.Run_accession.to_numpy()[eval_idx]
    off_eval = {t: torch.FloatTensor(offsets[t][eval_idx]).to(device) for t in task_names} if offsets is not None else None
    off_val = ({t: torch.FloatTensor(offsets[t][val_idx]).to(device) for t in task_names}
               if offsets is not None and val_idx is not None else None)
    X_val = torch.FloatTensor(X[val_idx]).to(device) if val_idx is not None else None
    y_val = {t: y[t][val_idx] for t in task_names} if val_idx is not None else None
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.lr_horizon or args.max_epochs)
                 if args.lr_schedule == "cosine" else None)
    ema_model = copy.deepcopy(model) if args.ema > 0 else None
    if args.lr is not None or scheduler is not None or ema_model is not None or args.early_stop != "none":
        logger.info("recipe: lr=%s schedule=%s ema=%s early_stop=%s", trainer.optimizer.param_groups[0]["lr"],
                    args.lr_schedule, args.ema or "off", args.early_stop)

    preds = {t: np.full((args.max_epochs, len(eval_runs)), -1, dtype=np.int16) for t in task_names}
    history = {"train_loss": [], "eval_macro_f1": {t: [] for t in task_names}, "lr": [],
               "val_macro_f1": [], "recipe": {"seed": seed, "lr": trainer.optimizer.param_groups[0]["lr"],
                                               "lr_schedule": args.lr_schedule, "ema": args.ema,
                                               "early_stop": args.early_stop, "val_split": args.val_split,
                                               "linear_init": str(args.linear_init_dir) if args.linear_init_dir else None,
                                               "study_weights": bool(args.study_weights), "depth_augment": args.depth_augment, "mixup": args.mixup, "adversarial": args.adversarial, "adversarial_mode": args.adversarial_mode,
                                               "patience": args.patience,
                                               "n_fit": int(len(fit_idx)), "n_val": int(len(val_idx)) if val_idx is not None else 0}}
    best_val, best_epoch, since_best, stopped_epoch = -1.0, None, 0, None
    pred_model = model
    for epoch in range(args.max_epochs):
        tr = trainer.train_one_epoch(train_loader)
        history["train_loss"].append(float(tr["loss"]))
        history["lr"].append(float(trainer.optimizer.param_groups[0]["lr"]))
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            with torch.no_grad():
                if epoch == 0:
                    ema_model.load_state_dict(model.state_dict())
                else:
                    for p_e, p_m in zip(ema_model.parameters(), model.parameters()):
                        p_e.mul_(args.ema).add_(p_m.detach(), alpha=1.0 - args.ema)
                    for b_e, b_m in zip(ema_model.buffers(), model.buffers()):
                        b_e.copy_(b_m)
            pred_model = ema_model
        ep = predict_argmax(pred_model, X_eval, task_names, off_eval)
        for t in task_names:
            preds[t][epoch] = ep[t]
            lab = y_eval[t] != IGNORE_INDEX
            f1 = (f1_score(y_eval[t][lab], ep[t][lab], average="macro", zero_division=0)
                  if lab.any() else float("nan"))
            history["eval_macro_f1"][t].append(float(f1))
        if X_val is not None:
            vp = predict_argmax(pred_model, X_val, task_names, off_val)
            scores = []
            for t in task_names:
                lab = y_val[t] != IGNORE_INDEX
                if lab.any():
                    scores.append(f1_score(y_val[t][lab], vp[t][lab], average="macro", zero_division=0))
            v = float(np.mean(scores)) if scores else float("nan")
            history["val_macro_f1"].append(v)
            if v > best_val:
                best_val, best_epoch, since_best = v, epoch, 0
            else:
                since_best += 1
            if since_best >= args.patience:
                stopped_epoch = epoch + 1
                logger.info("early stopping at epoch %d: best inner macro-F1 %.4f at epoch %d",
                            stopped_epoch, best_val, best_epoch + 1)
                break
        if (epoch + 1) % 25 == 0:
            logger.info("epoch %d/%d  train loss %.4f  eval macro-F1 %s", epoch + 1,
                        args.max_epochs, tr["loss"],
                        {t: round(history["eval_macro_f1"][t][-1], 3) for t in task_names})
    if X_val is not None:
        history["selected_epoch"] = int(best_epoch + 1) if best_epoch is not None else None
        history["stopped_epoch"] = int(stopped_epoch or args.max_epochs)
        history["best_val_macro_f1"] = float(best_val)
    if args.save_probs:
        pred_model.eval()
        probs = {}
        with torch.no_grad():
            for t in task_names:
                parts = []
                for i in range(0, len(X_eval), PREDICT_BATCH):
                    lg = pred_model(X_eval[i:i + PREDICT_BATCH])[t]
                    if off_eval is not None:
                        lg = lg + off_eval[t][i:i + PREDICT_BATCH]
                    parts.append(torch.softmax(lg, dim=1).cpu().numpy())
                probs[t] = np.concatenate(parts).astype(np.float32)
        np.savez_compressed(args.out / "probs_last.npz", **probs)

    np.savez_compressed(args.out / "preds_by_epoch.npz", **preds)
    (args.out / "eval_runs.txt").write_text("\n".join(eval_runs) + "\n")
    json.dump(classes, open(args.out / "label_classes.json", "w"), indent=2)
    json.dump(history, open(args.out / "history.json", "w"))
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
