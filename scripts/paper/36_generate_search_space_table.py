#!/usr/bin/env python3
"""
36_generate_search_space_table.py

Generate Supplementary Table 13: Optuna Hyperparameter Search Space

PURPOSE:
    Document every hyperparameter searched by Optuna during cross-validation,
    including type, range/choices, and the value selected in the final model.

INPUTS:
    - results/training_bioproject_v5/final_training_config.json  (best values)
    - configs/train_config_bioproject_v2.yaml                    (Optuna settings)

OUTPUTS:
    - paper/tables/final/sup_table_13_optuna_search_space.tex

USAGE:
    python scripts/paper/36_generate_search_space_table.py
"""

import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "paper" / "tables" / "final" / "sup_table_13_optuna_search_space.tex"


def load_best_values():
    cfg_path = REPO / "results" / "training_bioproject_v5" / "final_training_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    hp = cfg["hyperparameters"]
    mp = hp["model_params"]
    tp = hp["trainer_params"]
    batch_size = hp["batch_size"]

    # Flatten hidden dims as a string like "[282, 333, 320, 384]"
    hidden = mp["hidden_dims"]
    best = {
        "n_layers":                  len(hidden),
        "hidden_dim_per_layer":      str(hidden),
        "dropout":                   f"{mp['dropout']:.4f}",
        "activation":                mp["activation"],
        "use_batch_norm":            str(mp["use_batch_norm"]),
        "learning_rate":             f"{tp['learning_rate']:.2e}",
        "weight_decay":              f"{tp['weight_decay']:.2e}",
        "batch_size":                str(batch_size),
        "task_weight_sample_type":   f"{tp['task_weights']['sample_type']:.4f}",
        "task_weight_community":     f"{tp['task_weights']['community_type']:.4f}",
        "task_weight_host":          f"{tp['task_weights']['sample_host']:.4f}",
        "task_weight_material":      f"{tp['task_weights']['material']:.4f}",
        # Label smoothing was 0 in v5 (--no_label_smoothing flag)
        "ls_sample_type":            "0.0 (fixed)",
        "ls_community_type":         "0.0 (fixed)",
        "ls_sample_host":            "0.0 (fixed)",
        "ls_material":               "0.0 (fixed)",
    }
    return best


def generate_table(output_path: Path):
    best = load_best_values()

    # Each row: (Group, Parameter, Type, Range / Choices, Selected Value)
    rows = [
        # Architecture
        ("Architecture", "Number of hidden layers",
         "Integer", "2–4",
         best["n_layers"]),
        ("Architecture", "Hidden units per layer",
         "Integer", "64–512 (step 64), sampled independently per layer",
         best["hidden_dim_per_layer"]),
        ("Architecture", "Dropout rate",
         "Float", "0.10–0.50",
         best["dropout"]),
        ("Architecture", "Activation function",
         "Categorical", r"\{ReLU, GELU, Leaky ReLU\}",
         r"\texttt{leaky\_relu}"),
        ("Architecture", "Batch normalisation",
         "Categorical", r"\{True, False\}",
         best["use_batch_norm"]),
        # Optimisation
        ("Optimisation", "Learning rate",
         "Float (log)", r"$10^{-5}$–$10^{-2}$",
         best["learning_rate"]),
        ("Optimisation", "Weight decay",
         "Float (log)", r"$10^{-6}$–$10^{-3}$",
         best["weight_decay"]),
        ("Optimisation", "Batch size",
         "Categorical", r"\{32, 64, 128, 256\}",
         best["batch_size"]),
        # Task weights
        ("Task weights", "Sample type weight",
         "Float", "0.50–2.00",
         best["task_weight_sample_type"]),
        ("Task weights", "Community type weight",
         "Float", "0.50–2.00",
         best["task_weight_community"]),
        ("Task weights", "Sample host weight",
         "Float", "0.50–2.00",
         best["task_weight_host"]),
        ("Task weights", "Material weight",
         "Float", "0.50–2.00",
         best["task_weight_material"]),
        # Label smoothing
        ("Label smoothing", "Sample type $\\epsilon$",
         "Float", "0.00–0.15",
         best["ls_sample_type"]),
        ("Label smoothing", "Community type $\\epsilon$",
         "Float", "0.00–0.15",
         best["ls_community_type"]),
        ("Label smoothing", "Sample host $\\epsilon$",
         "Float", "0.00–0.15",
         best["ls_sample_host"]),
        ("Label smoothing", "Material $\\epsilon$",
         "Float", "0.00–0.15",
         best["ls_material"]),
    ]

    lines = []
    lines.append(r"\centering")
    lines.append(
        r"\caption{Optuna hyperparameter search space. "
        r"All 16 parameters were optimised jointly using Tree-structured Parzen Estimator (TPE) "
        r"Bayesian optimisation over 50 trials per outer cross-validation fold (5-fold CV, "
        r"3-fold inner CV). Label smoothing was fixed to 0 in the final v5 model "
        r"(\texttt{--no\_label\_smoothing}). "
        r"The selected values are those of the best-performing fold used for the final model."
        r"\label{tab:search_space}}"
    )
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llllr@{}}"
    )
    lines.append(r"\toprule")
    lines.append(r"Group & Parameter & Type & Range / Choices & Selected value \\")
    lines.append(r"\midrule")

    prev_group = None
    for group, param, ptype, search_range, selected in rows:
        if prev_group is not None and group != prev_group:
            lines.append(r"\addlinespace")
        group_cell = group if group != prev_group else ""
        lines.append(
            f"{group_cell} & {param} & {ptype} & {search_range} & {selected} \\\\"
        )
        prev_group = group

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ {output_path.name}")


if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING OPTUNA SEARCH SPACE TABLE (SUPPLEMENTARY TABLE 13)")
    print("=" * 80)
    print()
    try:
        generate_table(OUTPUT)
        print()
        print("=" * 80)
        print("✓ COMPLETE")
        print("=" * 80)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
