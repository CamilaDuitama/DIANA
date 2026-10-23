"""Supplementary table: label coverage of the split, per task.

Upper block: runs without a label on each side and held-out runs whose label never
occurs in training. Lower block: those unseen classes (each single-BioProject, so a
whole-BioProject split cannot put them on both sides). From data/splits_v17 metadata.
"""

import csv
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v17"
OUT = [ROOT / "results/paper_v2/sup_table_unseen_classes.tex",
       Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/tables/sup_table_unseen_classes.tex")]
TASKS = ["community_type", "feature", "sample_host", "material"]
TASK_LABEL = {"community_type": "Community type", "feature": "Feature",
              "sample_host": "Sample host", "material": "Material"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    train = load(SPLITS / "train_metadata.tsv")
    test = load(SPLITS / "test_metadata.tsv")

    summary_rows = []
    class_rows = []
    total_unseen_runs = 0
    for task in TASKS:
        tr_missing = sum(1 for r in train if not r.get(task, "").strip())
        te_missing = sum(1 for r in test if not r.get(task, "").strip())
        seen = {r[task] for r in train if r.get(task, "").strip()}
        held = [r for r in test if r.get(task, "").strip()]
        unseen_classes = sorted({r[task] for r in held} - seen)
        unseen_runs = sum(1 for r in held if r[task] not in seen)
        total_unseen_runs += unseen_runs
        cell = f"{unseen_runs} ({len(unseen_classes)} classes)" if unseen_runs else "0"
        summary_rows.append(
            f"{TASK_LABEL[task]} & {tr_missing:,} & {te_missing:,} & {cell} \\\\")
        for cls in unseen_classes:
            runs = [r for r in held if r[task] == cls]
            projects = len({r["archive_project"] for r in runs})
            class_rows.append(
                f"{TASK_LABEL[task]} & \\textit{{{cls}}} & {len(runs)} & {projects} \\\\")

    tex = f"""\\centering
\\caption{{Label coverage of the split, per task: runs without a label on each side
({len(train):,} training and {len(test):,} held-out runs in total), and held-out runs
whose label never occurs in training. A run without a label for a task is excluded from
that task's loss during training and from its evaluation. The lower block lists the
unseen classes: each occurs in a single BioProject, and whole BioProjects are assigned
to one side of the split, so these classes cannot appear on both sides. No classifier
can predict them; their {total_unseen_runs} runs are counted and reported, not scored
as errors.}}
\\label{{tab:sup_unseen_classes}}
\\begin{{tabular}}{{lrrr}}
\\toprule
Task & \\shortstack[r]{{Training runs\\\\without a label}} & \\shortstack[r]{{Held-out runs\\\\without a label}} & \\shortstack[r]{{Held-out runs with a label\\\\unseen in training}} \\\\
\\midrule
{chr(10).join(summary_rows)}
\\bottomrule
\\end{{tabular}}

\\vspace{{3mm}}

\\begin{{tabular}}{{llrr}}
\\toprule
Task & Class unseen in training & Held-out runs & BioProjects \\\\
\\midrule
{chr(10).join(class_rows)}
\\bottomrule
\\end{{tabular}}
"""
    for p in OUT:
        p.write_text(tex)
        log.info("wrote %s", p)


if __name__ == "__main__":
    main()
