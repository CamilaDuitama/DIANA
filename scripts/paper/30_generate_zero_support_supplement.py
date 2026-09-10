#!/usr/bin/env python3
"""
Supplementary Table: Classes with Zero Support in Test / Validation Sets

PURPOSE:
    For each task, list every class that is in the training vocabulary but has
    zero samples in the held-out test set and/or the held-out validation set.
    This table provides the context needed to interpret the F1† values in
    Main Table 1 (macro-averaged F1 restricted to seen classes).

INPUTS (from config.py):
    - PATHS['test_metrics']       : test_metrics.json (classification_report per task)
    - PATHS['label_encoders']     : label encoder class lists
    - PATHS['validation_metadata']: validation metadata TSV

OUTPUTS:
    - paper/tables/final/sup_table_12_zero_support_classes.tex

USAGE:
    python scripts/paper/30_generate_zero_support_supplement.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, SAMPLE_TYPE_MAP

sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions


TASK_PRETTY = {
    'sample_type':    'Sample Type',
    'community_type': 'Community Type',
    'sample_host':    'Sample Host',
    'material':       'Material',
}


# ── helpers ──────────────────────────────────────────────────────────────────

def tex_escape(s: str) -> str:
    return s.replace('_', '\\_').replace('&', '\\&').replace('%', '\\%')


def italicise_host(name: str) -> str:
    if name in ('Not applicable - env sample', 'Other mammal'):
        return tex_escape(name)
    return f"\\textit{{{tex_escape(name)}}}"


def get_test_supports(test_metrics: dict, label_encoders: dict) -> dict:
    """
    Returns {task: {class_name: support}} from the stored classification_report.
    Classes absent from the report (not predicted on, not in y_true) default to 0.
    """
    result = {}
    for task in TASKS:
        classes = label_encoders[task]['classes']
        cr = test_metrics[task]['classification_report']
        support_map = {classes[int(k)]: int(v['support'])
                       for k, v in cr.items()
                       if k not in ('accuracy', 'macro avg', 'weighted avg')
                       and k.lstrip('-').isdigit()
                       and int(k) < len(classes)}
        # Any class not in the report has support 0
        for c in classes:
            support_map.setdefault(c, 0)
        result[task] = support_map
    return result


def get_val_supports(label_encoders: dict) -> dict:
    """
    Loads validation predictions and returns {task: {class_name: support}}
    where support = number of validation samples with that true label
    (seen-label samples only, matching the filtering used for metrics).
    """
    metadata = pd.read_csv(PATHS['validation_metadata'], sep='\t')

    # Build a flat count per task from metadata directly (simpler than re-running loader)
    result = {}
    for task in TASKS:
        classes = label_encoders[task]['classes']
        col = metadata[task] if task in metadata.columns else pd.Series(dtype=str)

        # Normalise sample_type labels
        if task == 'sample_type':
            col = col.map(lambda x: SAMPLE_TYPE_MAP.get(x, x) if pd.notna(x) else x)
            known = set(SAMPLE_TYPE_MAP.get(c, c) for c in classes)
        else:
            known = set(classes)

        counts = col[col.isin(known)].value_counts().to_dict()
        if task == 'sample_type':
            # counts keyed by normalised name; re-key back to original class name
            support_map = {c: int(counts.get(SAMPLE_TYPE_MAP.get(c, c), 0)) for c in classes}
        else:
            support_map = {c: int(counts.get(c, 0)) for c in classes}
        result[task] = support_map
    return result


# ── main table builder ────────────────────────────────────────────────────────

def generate_table(output_dir: Path) -> None:
    print("\n[1/4] Loading label encoders...")
    with open(PATHS['label_encoders']) as f:
        label_encoders = json.load(f)

    print("\n[2/4] Loading test metrics...")
    with open(PATHS['test_metrics']) as f:
        test_metrics = json.load(f)

    print("\n[3/4] Computing per-class supports...")
    test_supports  = get_test_supports(test_metrics, label_encoders)
    val_supports   = get_val_supports(label_encoders)

    # Collect all classes (across all tasks) that are 0 in test OR 0 in val
    rows = []
    for task in TASKS:
        classes = label_encoders[task]['classes']
        for c in sorted(classes):
            t_sup = test_supports[task].get(c, 0)
            v_sup = val_supports[task].get(c, 0)
            rows.append({
                'task':      task,
                'class':     c,
                'test_sup':  t_sup,
                'val_sup':   v_sup,
                'any_zero':  (t_sup == 0 or v_sup == 0),
            })

    df = pd.DataFrame(rows)
    df_zero = df[df['any_zero']].copy()

    print(f"  Classes with 0 support in test or val: {len(df_zero)}")

    # ── LaTeX ──────────────────────────────────────────────────────────────
    print("\n[4/4] Writing LaTeX table...")

    n_test  = test_metrics[TASKS[0]]['n_samples']
    n_val   = len(pd.read_csv(PATHS['validation_metadata'], sep='\t'))

    lines = [
        r"\centering",
        r"\caption{Training-vocabulary classes absent from the held-out test set "
        r"(n\,=\," + str(n_test) + r") and/or the validation set "
        r"(n\,=\," + str(n_val) + r"). "
        r"For each class the number of samples in each split is shown. "
        r"Classes with zero support contribute F1\,=\,0 under standard macro-averaging; "
        r"the F1$^{\dagger}$ values in Table~\ref{tab:performance} exclude these "
        r"classes to give a fair reflection of per-class accuracy on actually observed labels.}",
        r"\label{tab:zero_support}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Task & Class & Test ($n$) & Validation ($n$) \\",
        r"\midrule",
    ]

    current_task = None
    for _, row in df_zero.sort_values(['task', 'class']).iterrows():
        task = row['task']
        cls  = row['class']
        t    = int(row['test_sup'])
        v    = int(row['val_sup'])

        if task != current_task:
            if current_task is not None:
                lines.append(r"\addlinespace")
            current_task = task
            task_str = f"\\textbf{{{TASK_PRETTY[task]}}}"
        else:
            task_str = ""

        if task == 'sample_host':
            cls_str = italicise_host(cls)
        else:
            cls_str = tex_escape(cls)

        # Mark zero cells
        t_str = f"\\textbf{{0}}" if t == 0 else str(t)
        v_str = f"\\textbf{{0}}" if v == 0 else str(v)

        lines.append(f"{task_str} & {cls_str} & {t_str} & {v_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2mm]",
        r"{\footnotesize Bold zeros indicate the class is completely absent from that split. "
        r"Classes present in both splits are not shown.}",
    ]

    out = output_dir / "sup_table_12_zero_support_classes.tex"
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ {out}")

    # ── also print a quick summary to stdout ──────────────────────────────
    print("\nZero-support class summary:")
    print(f"{'Task':<18} {'Class':<35} {'Test':>6} {'Val':>6}")
    print("-" * 70)
    for _, row in df_zero.sort_values(['task', 'class']).iterrows():
        marker = lambda n: str(n) if n > 0 else "ZERO"
        print(f"{row['task']:<18} {row['class']:<35} {marker(row['test_sup']):>6} {marker(row['val_sup']):>6}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SUPPLEMENTARY TABLE — ZERO-SUPPORT CLASSES")
    print("=" * 70)
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_table(output_dir)
    print("\n" + "=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
