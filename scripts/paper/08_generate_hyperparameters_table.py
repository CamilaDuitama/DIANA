#!/usr/bin/env python3
"""
Generate Hyperparameters Table (Supplementary Table 4)

PURPOSE:
    Show optimized model hyperparameters from cross-validation.

INPUTS (from config.py):
    - PATHS['cv_results'] → results/training/cv_results/best_hyperparameters.json
      Contains aggregated hyperparameters from 5-fold CV with Optuna

OUTPUTS:
    - paper/tables/final/sup_table_04_hyperparameters.tex

PROCESS:
    1. Load best_hyperparameters.json
    2. Extract architecture parameters (layers, dropout, activation)
    3. Extract training parameters (learning rate, weight decay, batch size)
    4. Extract task weights (sample_type, community_type, sample_host, material)
    5. Format as LaTeX table with three categories

CONFIGURATION:
    All paths imported from config.py

HARDCODED VALUES:
    - Input features: 107,480 (total unitigs in matrix)
    - Max epochs: 200 (from configs/train_config.yaml)
    - CV folds: 5 (standard 5-fold cross-validation)
    - Optuna trials: 50 per fold (from configs/train_config.yaml)
    - Category groupings: Architecture, Training, Task Weights

DEPENDENCIES:
    - json, pathlib
    - config.py

USAGE:
    python scripts/paper/generate_hyperparameters_table.py
    
AUTHOR: Refactored from 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_hyperparameters_table(output_dir):
    """Generate table with optimized hyperparameters."""
    print("\n[1/2] Loading hyperparameters...")
    hyperparams_file = Path(PATHS['hyperparameters'])
    
    with open(hyperparams_file) as f:
        params = json.load(f)
    
    print(f"  ✓ Loaded from {hyperparams_file}")
    
    print("\n[2/2] Generating LaTeX table...")
    
    lines = []
    lines.append("\\centering")
    lines.append("\\caption{Optimized model hyperparameters\\label{tab:hyperparameters}}")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lll@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Category & Parameter & Value \\\\")
    lines.append("\\midrule")
    
    # Architecture
    lines.append("\\multirow{4}{*}{Architecture} & Input features & 107,480 \\\\")
    
    # Build hidden layers list
    n_layers = int(params['n_layers'])
    hidden_dims = [int(params[f'hidden_dim_{i}']) for i in range(n_layers)]
    hidden_str = str(hidden_dims).replace('[', '{[}').replace(']', '{]}')
    lines.append(f" & Hidden layers & {hidden_str} \\\\")
    lines.append(f" & Dropout rate & {params['dropout']:.4f} \\\\")
    
    # Escape activation function name
    activation_str = params['activation'].replace('_', '\\_')
    lines.append(f" & Activation & {activation_str} \\\\")
    lines.append("\\addlinespace")
    
    # Training
    lines.append("\\multirow{{4}}{{*}}{{Training}} & Learning rate & {:.6f} \\\\".format(params['learning_rate']))
    lines.append(" & Weight decay & {:.2e} \\\\".format(params['weight_decay']))
    lines.append(f" & Batch size & {int(params['batch_size'])} \\\\")
    lines.append(" & Max epochs & 200 \\\\")
    lines.append("\\addlinespace")
    
    # Task weights
    lines.append("\\multirow{{4}}{{*}}{{Task Weights}} & Sample Type & {:.3f} \\\\".format(params['task_weight_sample_type']))
    lines.append(" & Community Type & {:.3f} \\\\".format(params['task_weight_community']))
    lines.append(" & Sample Host & {:.3f} \\\\".format(params['task_weight_host']))
    lines.append(" & Material & {:.3f} \\\\".format(params['task_weight_material']))
    
    lines.append("\\botrule")
    lines.append("\\end{tabular*}")
    lines.append("\\\\[2mm]")
    
    footnote_parts = [
        "Hyperparameters determined via 5-fold cross-validation with 50 Optuna trials per fold.",
        "Values shown are aggregated from best trials across folds (mean for numeric, mode for categorical).",
        "Input features: 107,480 unitigs from k-mer matrix (muset).",
        "Max epochs: 200 with early stopping based on validation loss."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    output_file = output_dir / "sup_table_04_hyperparameters.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING HYPERPARAMETERS TABLE (SUP TABLE 4)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_hyperparameters_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Hyperparameters table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
