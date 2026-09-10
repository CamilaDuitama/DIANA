#!/usr/bin/env python3
"""What does each shared-trunk dimension separate? And is any of it just depth?

Two jobs in one pass, because both need the same trunk activations.

1. **Interpretation (R1.1).** The trunk is what the four heads share, so if
   multi-task learning helps (it does -- Table 8), the shared representation is the
   thing to describe. For every dimension, measure how much of its variance is
   explained by class membership in each task (eta-squared from a one-way ANOVA:
   0 = the dimension ignores that task, 1 = it separates the classes perfectly).

2. **Confounder check (R3.11, R3.7/R3.8).** The same dimensions are correlated
   against sequencing depth and library length. A dimension that tracks depth more
   strongly than any label is encoding how much was sequenced, not what was
   sequenced -- and if such dimensions dominate the trunk, the depth-only control
   (Table 7) understates the problem. This is the honest place for that check,
   because it looks inside the model rather than at its outputs.

Depth is measured from the data, not from SRA bookkeeping: total unitig abundance
per sample, and mean k-mer fraction. Library length comes from AMD when present.

    ./env/bin/python scripts/analysis/13_characterise_trunk_dims.py \\
        --model results/search_v9_final/final_model/best_model.pth \\
        --config results/search_v9_final/final_training_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = PROJECT_ROOT / "data/matrices/matrix_v9_train"
AMD = PROJECT_ROOT / "data/metadata/AncientMetagenomeDir-v26.03.0"


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of a dimension's variance explained by group membership."""
    ok = pd.notna(labels)
    v, l = values[ok], labels[ok]
    groups = [v[l == g] for g in pd.unique(l)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    grand = v.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((v - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/trunk_dims_v9")
    args = ap.parse_args()

    from diana.data.loader import MatrixLoader
    from diana.models.multitask_mlp import MultiTaskMLP

    cfg = json.loads(args.config.read_text())
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    hp = cfg["hyperparameters"]
    tasks = cfg["task_names"]

    feats, ids, _ = MatrixLoader(Path(cfg["features_path"])).load()
    md = (pd.read_csv(cfg["metadata_path"], sep="\t", low_memory=False)
          .set_index("Run_accession").loc[list(ids)].reset_index())

    n_cls = {t: sd[f"heads.{t}.3.weight"].shape[0] for t in tasks}
    model = MultiTaskMLP(input_dim=feats.shape[1],
                         hidden_dims=hp["model_params"]["hidden_dims"],
                         num_classes=n_cls, regression_tasks=[],
                         dropout=hp["model_params"]["dropout"],
                         use_batch_norm=hp["model_params"]["use_batch_norm"],
                         activation=hp["model_params"]["activation"])
    model.load_state_dict(sd)
    model.eval().to(args.device)

    # Trunk activations: the representation the heads share.
    acts = []
    with torch.no_grad():
        for i in range(0, len(feats), 256):
            xb = torch.tensor(feats[i:i + 256], dtype=torch.float32, device=args.device)
            acts.append(model.backbone(xb).cpu().numpy())
    A = np.concatenate(acts)
    n_dim = A.shape[1]
    print(f"trunk: {A.shape[0]} samples x {n_dim} dimensions "
          f"(width comes from the search, not hard-coded)")

    # --- depth, measured from the data ---
    Ab, _, _ = MatrixLoader(MATRIX_DIR / "unitigs.abundance.mat").load()
    depth = pd.DataFrame({
        "log_abundance": np.log10(np.clip(Ab.sum(axis=1), 1, None)),
        "mean_fraction": feats.mean(axis=1),
    })
    del Ab

    # --- library length from AMD, when recorded ---
    lib_len = pd.Series(np.nan, index=range(len(md)))
    try:
        lib = pd.concat([pd.read_csv(AMD / f"ancientmetagenome-{k}_libraries.tsv",
                                     sep="\t", low_memory=False)
                         for k in ("hostassociated", "environmental")],
                        ignore_index=True).drop_duplicates("archive_data_accession")
        cand = [c for c in lib.columns if "read_length" in c or "length" in c.lower()]
        if cand:
            m = dict(zip(lib.archive_data_accession,
                         pd.to_numeric(lib[cand[0]], errors="coerce")))
            lib_len = md.Run_accession.map(m)
            print(f"library length from AMD column '{cand[0]}': "
                  f"{int(lib_len.notna().sum())}/{len(md)} runs")
    except Exception as e:
        print(f"library length unavailable ({e})")

    rows = []
    for d in range(n_dim):
        v = A[:, d]
        rec = {"dim": d, "activation_sd": float(v.std())}
        for t in tasks:
            rec[f"eta2_{t}"] = eta_squared(v, md[t].to_numpy())
        for name, col in depth.items():
            rho, p = stats.spearmanr(v, col.to_numpy())
            rec[f"rho_{name}"] = float(rho)
            rec[f"p_{name}"] = float(p)
        if lib_len.notna().sum() > 20:
            ok = lib_len.notna().to_numpy()
            rho, p = stats.spearmanr(v[ok], lib_len[ok].to_numpy())
            rec["rho_lib_length"] = float(rho)
            rec["p_lib_length"] = float(p)
        rows.append(rec)

    df = pd.DataFrame(rows)
    eta_cols = [f"eta2_{t}" for t in tasks]
    df["best_task"] = df[eta_cols].idxmax(axis=1).str.replace("eta2_", "", regex=False)
    df["best_eta2"] = df[eta_cols].max(axis=1)
    df["max_abs_rho_depth"] = df[["rho_log_abundance", "rho_mean_fraction"]].abs().max(axis=1)
    # The confounder test: does depth explain this dimension better than any label?
    df["depth_dominated"] = df.max_abs_rho_depth ** 2 > df.best_eta2

    args.output.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output / "trunk_dims.tsv", sep="\t", index=False)

    n_dd = int(df.depth_dominated.sum())
    lines = [f"Shared-trunk dimensions — {n_dim} of them", ""]
    lines.append("eta2 = fraction of a dimension's variance explained by that task's")
    lines.append("class membership. rho = Spearman against a depth measure.")
    lines.append("")
    lines.append("Dimensions primarily separating each task:")
    lines.append(df.groupby("best_task").agg(
        n_dims=("dim", "size"), mean_eta2=("best_eta2", "mean"),
        max_eta2=("best_eta2", "max")).to_string())
    lines.append("")
    lines.append(f"CONFOUNDER CHECK: {n_dd}/{n_dim} dimensions "
                 f"({100 * n_dd / n_dim:.1f} %) are better explained by sequencing "
                 f"depth than by any label.")
    lines.append(f"  median |rho| with depth : {df.max_abs_rho_depth.median():.3f}")
    lines.append(f"  median best eta2        : {df.best_eta2.median():.3f}")
    if n_dd > n_dim / 2:
        lines.append("  >> More than half the trunk tracks depth. Report this "
                     "prominently; it qualifies the depth-only control.")
    else:
        lines.append("  >> Depth does not dominate the representation, which is "
                     "consistent with the depth-only control (Table 7).")
    lines.append("")
    lines.append("Top 10 most class-discriminative dimensions:")
    lines.append(df.nlargest(10, "best_eta2")[
        ["dim", "best_task", "best_eta2", "max_abs_rho_depth", "depth_dominated"]
    ].to_string(index=False))

    report = "\n".join(lines)
    print("\n" + report)
    (args.output / "summary.txt").write_text(report + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
