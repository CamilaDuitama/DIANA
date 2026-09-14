#!/usr/bin/env python3
"""D3: per-regime detector thresholds against one global threshold.

A single cut on `1 - P(stated label)` misfires on the tail, because the model is
far less confident about rare classes: mean P(correct label) is 0.025 for
few-shot `material` against 0.495 many-shot. So the same score means different
things depending on how well represented the stated class is, and one threshold
cannot be right for both.

Protocol, fixed before the numbers were seen
--------------------------------------------
Thresholds are set on the 2,716 **out-of-fold training** probabilities from C1,
never on held-out. Each held-back dev fold was predicted by a model that never
saw its BioProject, and training labels are correct, so every out-of-fold run is
a clean negative and the false-flag rate is directly measurable there.

* global: one threshold per task, firing on 5 % of all out-of-fold runs.
* per-regime: one threshold per (task, regime), firing on 5 % of the out-of-fold
  runs whose **true** class falls in that regime.

Held-out runs are assigned to the regime of their **stated** label, because that
is the only thing known at inference time. This differs deliberately from the
regime table in section 2, which stratifies by true class as a scientific
breakdown rather than an operating rule.

Adopt per-regime thresholds only if overall recall rises while the overall
false-flag rate stays at or below 6 %.

This scores the held-out set, so it is an additional held-out read and is
recorded as one.

    ./env/bin/python scripts/analysis/24_per_regime_thresholds.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v9"
CAL = ROOT / "results/calibration_v9"
EVAL = ROOT / "results/final_eval_v9"
PLANTED = ROOT / "results/planted_mislabels_v9/planted_test_mixed_r0.1.tsv"
OUT = ROOT / "results/per_regime_thresholds"

TASKS = ["community_type", "feature", "sample_host", "material"]
REGIMES = [("few-shot (<20)", 0, 20), ("medium-shot (20-100)", 20, 100),
           ("many-shot (>100)", 100, np.inf)]
BUDGET = 0.05
FALSE_FLAG_CEILING = 0.06


def regime_of(n: float) -> str | None:
    for name, lo, hi in REGIMES:
        if lo < n <= hi:
            return name
    return None


def support_map(task: str) -> dict[str, str]:
    train = pd.read_csv(SPLITS / "train_metadata.tsv", sep="\t", low_memory=False)
    counts = train[task].value_counts()
    return {str(c): r for c, n in counts.items() if (r := regime_of(n))}


def p_of_label(pred: pd.DataFrame, task: str, labels: pd.Series) -> np.ndarray:
    cols = sorted((c for c in pred.columns if c.startswith(f"{task}_prob_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = pred[cols].to_numpy()
    name_by_idx = (pred[[f"{task}_true_idx", f"{task}_true"]].dropna().drop_duplicates()
                   .set_index(f"{task}_true_idx")[f"{task}_true"].to_dict())
    idx = {str(v): int(k) for k, v in name_by_idx.items()}
    out = np.full(len(pred), np.nan)
    for i, lab in enumerate(labels.astype(str).to_numpy()):
        j = idx.get(lab)
        if j is not None and j < P.shape[1]:
            out[i] = P[i, j]
    return out


def load_oof(task: str) -> pd.DataFrame:
    frames = []
    for f in sorted(CAL.glob(f"oof_{task}_fold*")):
        p = f / "test_predictions.tsv"
        if p.exists():
            frames.append(pd.read_csv(p, sep="\t"))
    if not frames:
        raise FileNotFoundError(f"no out-of-fold predictions for {task}")
    return frames


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    plant = pd.read_csv(PLANTED, sep="\t")
    rows, thr_rows = [], []

    for task in TASKS:
        reg = support_map(task)

        # --- thresholds from out-of-fold training runs (all clean negatives) ---
        scores, regs = [], []
        for fr in load_oof(task):
            lab = fr[f"{task}_true"]
            ok = lab.notna()
            s = 1.0 - p_of_label(fr[ok], task, lab[ok])
            r = lab[ok].astype(str).map(reg).to_numpy()
            good = ~np.isnan(s)
            scores.append(s[good]); regs.append(r[good])
        s_oof = np.concatenate(scores); r_oof = np.concatenate(regs)
        t_global = float(np.quantile(s_oof, 1.0 - BUDGET))
        t_regime = {}
        for name, _, _ in REGIMES:
            sel = r_oof == name
            if sel.sum() >= 20:
                t_regime[name] = float(np.quantile(s_oof[sel], 1.0 - BUDGET))
        thr_rows.append({"task": task, "n_oof": int(len(s_oof)),
                         "t_global": t_global,
                         **{f"t_{n.split()[0]}": t_regime.get(n, np.nan)
                            for n, _, _ in REGIMES}})

        # --- apply to held-out ---
        dp = pd.read_csv(EVAL / f"heldout_single_{task}/test_predictions.tsv", sep="\t")
        keep = plant[["Run_accession", task, f"{task}_planted"]]
        m = dp.merge(keep, on="Run_accession")
        m = m[m[task].notna() & m[f"{task}_planted"].notna()]
        s = 1.0 - p_of_label(m, task, m[task])
        ok = ~np.isnan(s)
        s = s[ok]
        y = m.loc[ok, f"{task}_planted"].astype(bool).to_numpy()
        stated_reg = m.loc[ok, task].astype(str).map(reg).to_numpy()

        for scheme in ("global", "per-regime"):
            if scheme == "global":
                fired = s > t_global
            else:
                fired = np.zeros(len(s), bool)
                for name in set(stated_reg[pd.notna(stated_reg)]):
                    t = t_regime.get(name, t_global)
                    sel = stated_reg == name
                    fired[sel] = s[sel] > t
                unknown = pd.isna(stated_reg)
                fired[unknown] = s[unknown] > t_global
            rows.append({
                "task": task, "scheme": scheme, "n_scored": int(len(s)),
                "n_planted": int(y.sum()),
                "recall": float(fired[y].mean()) if y.sum() else np.nan,
                "false_flag_rate": float(fired[~y].mean()),
                "n_flagged": int(fired.sum()),
            })

    res = pd.DataFrame(rows)
    thr = pd.DataFrame(thr_rows)
    res.to_csv(OUT / "comparison.tsv", sep="\t", index=False)
    thr.to_csv(OUT / "thresholds.tsv", sep="\t", index=False)

    print("\nthresholds set on out-of-fold training runs:")
    print(thr.round(4).to_string(index=False))
    print("\nheld-out result:")
    print(res.round(4).to_string(index=False))

    print("\nper task, per-regime minus global:")
    verdicts = []
    for task in TASKS:
        g = res[(res.task == task) & (res.scheme == "global")].iloc[0]
        r = res[(res.task == task) & (res.scheme == "per-regime")].iloc[0]
        adopt = (r.recall > g.recall) and (r.false_flag_rate <= FALSE_FLAG_CEILING)
        verdicts.append(adopt)
        print(f"  {task:16s} recall {g.recall:.3f} -> {r.recall:.3f} "
              f"({r.recall-g.recall:+.3f})   false-flag {g.false_flag_rate:.3f} -> "
              f"{r.false_flag_rate:.3f}   adopt={adopt}")
    print(f"\npre-committed rule (recall up AND false-flag <= {FALSE_FLAG_CEILING}): "
          f"adopt on {sum(verdicts)} of {len(verdicts)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
