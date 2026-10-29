"""Supplementary table: label coverage of the split, per task.

Upper block: held-out runs without a label and held-out runs whose label never occurs in
training (training runs without a label are not shown: most runs carry either a sample host or
a feature, not both). Lower block: those unseen classes, with a note naming the training labels of the same species
(same first two words), or stating that no training label shares the genus. From data/splits_v19.
"""

import csv
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/splits_v19"
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


def close_labels(cls: str, seen: set[str]) -> str:
    """Note for an unseen class: training labels of the same species, else whether any shares the genus."""
    same = sorted(c for c in seen if c.split()[:2] == cls.split()[:2])
    if same:
        return "training has " + " and ".join(f"\\textit{{{c}}}" for c in same)
    genus = sorted(c for c in seen if c.split()[0] == cls.split()[0])
    return "training has " + " and ".join(f"\\textit{{{c}}}" for c in genus) if genus else "no training label of the same genus"


def main() -> None:
    train = load(SPLITS / "train_metadata.tsv")
    test = load(SPLITS / "test_metadata.tsv")

    summary_rows = []
    class_rows = []
    notes = []
    total_unseen_runs = 0
    both = sum(1 for r in train if r.get("feature", "").strip() and r.get("sample_host", "").strip())
    for task in TASKS:
        te_missing = sum(1 for r in test if not r.get(task, "").strip())
        seen = {r[task] for r in train if r.get(task, "").strip()}
        held = [r for r in test if r.get(task, "").strip()]
        unseen_classes = sorted({r[task] for r in held} - seen)
        unseen_runs = sum(1 for r in held if r[task] not in seen)
        total_unseen_runs += unseen_runs
        cell = f"{unseen_runs} ({len(unseen_classes)} classes)" if unseen_runs else "0"
        summary_rows.append(
            f"{TASK_LABEL[task]} & {te_missing:,} & {cell} \\\\")
        for cls in unseen_classes:
            runs = [r for r in held if r[task] == cls]
            projects = len({r["archive_project"] for r in runs})
            class_rows.append(
                f"{TASK_LABEL[task]} & \\textit{{{cls}}} & {len(runs)} & {projects} \\\\")
            notes.append(f"\\textit{{{cls}}}: {close_labels(cls, seen)}")

    tex = f"""\\centering
\\caption{{Held-out runs ({len(test):,} in total) without a label, and with a label that does not occur in training, per field. Most runs
have either a sample host or a feature, not both ({both} of the {len(train):,} training runs have both), so many runs lack a label for one of
these fields. The lower block lists the labels absent from training; each comes from a single BioProject.}}
\\label{{tab:sup_unseen_classes}}
\\begin{{tabular}}{{lrr}}
\\toprule
Field & \\shortstack[r]{{Held-out runs\\\\without a label}} & \\shortstack[r]{{Held-out runs with a label\\\\absent from training}} \\\\
\\midrule
{chr(10).join(summary_rows)}
\\bottomrule
\\end{{tabular}}

\\vspace{{3mm}}

\\begin{{tabular}}{{llrr}}
\\toprule
Field & Label absent from training & Held-out runs & BioProjects \\\\
\\midrule
{chr(10).join(class_rows)}
\\bottomrule
\\end{{tabular}}

\\vspace{{1mm}}
{{\\footnotesize Similar labels in training. {"; ".join(notes)}.\\par}}
"""
    for p in OUT:
        p.write_text(tex)
        log.info("wrote %s", p)


if __name__ == "__main__":
    main()
