#!/usr/bin/env python3
"""v19 held-out predictions of one tuned baseline for one task (held-out read 1, ledger).

The model and its fixed arguments come from 40_baselines_v19.py (`build`), its tuned
parameters from results/baselines_v19/<task>/best_<model>.json; it is fitted on every
labelled training run and predicts the held-out runs of results/v19_heldout/heldout_frac.npz
(same unitig order as the training matrix, verified by the projection gate). Writes, under
results/heldout_v19/baselines/<task>/: pred_<model>.tsv (Run_accession, y_true, y_pred)
and, when the model has probabilities, probs_<model>.tsv (Run_accession, y_true, p_<class>,
classes absent from training absent from the columns). No score is computed here.
With --on-train the fitted model predicts the training runs themselves instead (training-set
evaluation), written under results/train_fit_v19/baselines/<task>/. --matrix, --heldout-matrix,
--tuned and --out run the same steps on another input (the hypervector test).

    ./env/bin/python scripts/evaluation/44_baselines_heldout_v19.py --task feature --model kNN
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data/splits_v19"
MATRIX = ROOT / "data/matrices/matrix_v17_train/unitigs.frac.mat"
HELDOUT = ROOT / "results/v19_heldout/heldout_frac.npz"
TUNED = ROOT / "results/baselines_v19"
OUT = ROOT / "results/heldout_v19/baselines"
TRAIN_OUT = ROOT / "results/train_fit_v19/baselines"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def tuning_module():
    spec = importlib.util.spec_from_file_location("base40", ROOT / "scripts/evaluation/40_baselines_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--on-train", action="store_true", help="predict the training runs the model was fitted on")
    ap.add_argument("--matrix", type=Path, default=MATRIX, help="training input")
    ap.add_argument("--heldout-matrix", type=Path, default=HELDOUT, help="held-out input (npz: frac, sample_ids, feature_ids)")
    ap.add_argument("--tuned", type=Path, default=TUNED, help="folder of best_<model>.json per task")
    ap.add_argument("--out", type=Path, default=None, help="output root (default by mode)")
    ap.add_argument("--save-model", action="store_true", help="also save the fitted model (<out>/<task>/model_<model>.joblib) for diana-predict")
    a = ap.parse_args()
    T = tuning_module()
    if a.model not in T.GRIDS:
        raise SystemExit(f"{a.model} is not a v19 baseline")
    best = json.load(open(a.tuned / a.task / f"best_{a.model}.json"))
    if best["task"] != a.task or best["model"] != a.model:
        raise SystemExit("tuned-config file does not match the requested task/model")
    if best.get("study_weights"):
        raise SystemExit("this selection was made with study weights; refitting without them would not reproduce it")
    out = (a.out or (TRAIN_OUT if a.on_train else OUT)) / a.task
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"pred_{a.model}.tsv").exists():
        raise SystemExit(f"{out}/pred_{a.model}.tsv exists; refusing to overwrite")
    X, meta = T.load(SPLIT, a.matrix)
    lab = meta[a.task].replace("", np.nan).notna().to_numpy()
    if a.on_train:
        Xh, ids = X, meta.Run_accession.astype(str).to_numpy()
        y_true = meta[a.task].replace("", np.nan).to_numpy()
    else:
        with np.load(a.heldout_matrix, allow_pickle=False) as z:
            Xh, ids = z["frac"].astype(np.float32), z["sample_ids"].astype(str)
            if z["feature_ids"].shape[0] != X.shape[1]:
                raise SystemExit("held-out matrix width differs from the training matrix")
        test = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).set_index("Run_accession")
        if set(ids) != set(test.index):
            raise SystemExit("held-out matrix and test metadata hold different runs")
        y_true = test.loc[ids, a.task].replace("", np.nan).to_numpy()
    m = T.build(a.model, best["params"], a.n_jobs).fit(X[lab], meta.loc[lab, a.task].to_numpy())
    pd.DataFrame({"Run_accession": ids, "y_true": y_true, "y_pred": m.predict(Xh)}).to_csv(
        out / f"pred_{a.model}.tsv", sep="\t", index=False)
    if a.model not in T.NO_PROBA:
        P = pd.DataFrame(m.predict_proba(Xh), columns=[f"p_{c}" for c in m.classes_])
        P.insert(0, "y_true", y_true)
        P.insert(0, "Run_accession", ids)
        P.to_csv(out / f"probs_{a.model}.tsv", sep="\t", index=False)
    if a.save_model:
        import joblib
        joblib.dump(m, out / f"model_{a.model}.joblib")
    log.info("%s / %s: fitted on %d labelled training runs with %s, predicted %d %s runs",
             a.task, a.model, int(lab.sum()), best["params"], len(ids), "training" if a.on_train else "held-out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
