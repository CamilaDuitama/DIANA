#!/usr/bin/env python3
"""Supplementary tables of the dev-fold screens on the current split (PROTOCOLS.md, v19 amendments of 2026-10-21/22).

From results/screens_v19/verdicts.tsv (72_screen_verdicts_v19.py) and, for equal weight per study, the
re-scored dev-fold tables (results/rescore_v19/dev_single_studyweight/, results/classification_v19/dev_studyweight/),
classified with the same rule (72_ verdict). Each cell: paired difference in the share of planted mislabels
caught (mean over the budgets both models can operate at) / paired difference in macro-F1, variant minus
reference; ^- marks a loss and ^+ a win under the rule. Writes, in the manuscript repository:
  tables/sup_table_representations.tex  input representations (DIANA, then logistic regression)
  tables/sup_table_training_changes.tex training changes (DIANA)

    ./env/bin/python scripts/paper/generate_sup_tables_screens_v19.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/tables")
TASKS = ["community_type", "feature", "sample_host", "material"]
HEAD = "Community type & Feature & Sample host & Material"
INPUTS = {
    "pa": "presence/absence", "abund": "mean $k$-mer abundance (log)",
    "frac_pca192": "fraction, PCA-192", "frac_pca191nd": "fraction, PCA-191 (depth axis removed)",
    "pa_pca192": "presence/absence, PCA-192", "pa_pca191nd": "presence/absence, PCA-191 (depth axis removed)",
    "frac_kmer4": "fraction + 4-mer composition", "frac_clust": "fraction + sums over similar unitigs",
    "logdepth": "fraction + sequencing depth",
}
CHANGES = {
    "stop_random": "early stopping on a validation slice", "studyweight": "equal weight per BioProject",
    "lr_init": "network as a correction to logistic regression", "lr_offsets": "logistic-regression scores added to the network's",
    "ema": "weight averaging over epochs", "depth_aug": "random thinning to lower depth", "mixup": "mixup of runs sharing every label",
    "adv_conf0.1": "study-adversarial training, strength 0.1", "adv_conf0.3": "study-adversarial training, strength 0.3",
    "adv_conf1.0": "study-adversarial training, strength 1.0",
}
RULE = ("A variant wins a field ($^{+}$) only if it catches more planted mislabels (paired 95\\,\\% interval over BioProjects above zero, "
        "2,000 draws) and its macro-F1 does not clearly drop; it loses ($^{-}$) if it catches fewer, or if its macro-F1 interval lies entirely "
        "below zero with a drop of at least the field's seed-noise threshold (0.080, 0.042, 0.028 and 0.017 for community type, feature, "
        "sample host and material); all other cells are ties.")


def verdict_rule():
    spec = importlib.util.spec_from_file_location("v72", ROOT / "scripts/analysis/72_screen_verdicts_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verdict


def studyweight_rows() -> pd.DataFrame:
    f = pd.read_csv(ROOT / "results/rescore_v19/dev_single_studyweight/flagging_paired.tsv", sep="\t")
    c = pd.read_csv(ROOT / "results/classification_v19/dev_studyweight/classification_paired.tsv", sep="\t")
    m = c.merge(f, on=["task", "a", "b"], suffixes=("_f1", "_flag"), validate="one_to_one")
    m = m[(m.a == "DIANAW") & (m.b == "DIANA")]
    d = pd.DataFrame({"arm": "studyweight", "task": m.task, "b": "DIANA", "flag_delta": m.delta_choosing,
                      "flag_low": m.ci_low_flag, "flag_high": m.ci_high_flag, "flag_verdict": m.verdict_flag,
                      "f1_delta": m.delta, "f1_low": m.ci_low_f1, "f1_high": m.ci_high_f1})
    d["rule"] = d.apply(verdict_rule(), axis=1)
    return d


def num(x: float) -> str:
    s = f"{x:+.2f}"
    return "0.00" if s in ("+0.00", "-0.00") else s


def cell(r) -> str:
    mark = {"win": "^{+}", "loss": "^{-}"}.get(r.rule, "")
    return f"${num(r.flag_delta)}$ / ${num(r.f1_delta)}{mark}$"


def rows(v: pd.DataFrame, labels: dict, b: str) -> list[str]:
    out = []
    for arm, label in labels.items():
        d = v[(v.arm == arm) & (v.b == b)].set_index("task")
        if set(d.index) != set(TASKS):
            raise ValueError(f"{arm} vs {b}: fields {sorted(d.index)}")
        out.append(f"{label} & " + " & ".join(cell(d.loc[t]) for t in TASKS) + " \\\\")
    return out


def table(label: str, caption: str, blocks: list[tuple[str | None, list[str]]]) -> str:
    body = []
    for title, lines in blocks:
        if title:
            body += ["\\midrule", f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\"]
        else:
            body.append("\\midrule")
        body += lines
    return "\n".join(["\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
                      "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lcccc@{}}", "\\toprule", f"Variant & {HEAD} \\\\",
                      *body, "\\bottomrule", "\\end{tabular*}", ""])


def main() -> None:
    v = pd.read_csv(ROOT / "results/screens_v19/verdicts.tsv", sep="\t")
    rep = table("tab:sup_representations",
                "Input representations tested against the unitig fraction on the dev folds. Each cell gives the paired difference, "
                "alternative minus fraction, in the share of planted mislabels caught (mean over the 5, 10 and 20\\,\\% false-flag budgets "
                "both models can operate at) and in macro-F1. Upper block: \\diana{} with only its input changed (network and, for community "
                "type, its logistic-regression partner); lower block: logistic regression alone. Projections and scalings were fitted on the "
                "training folds of each dev fold only. " + RULE,
                [("\\diana{}", rows(v, INPUTS, "DIANA")), ("logistic regression", rows(v, {k: x for k, x in INPUTS.items()}, "LogReg"))])
    tc = pd.concat([v, studyweight_rows()], ignore_index=True)
    trn = table("tab:sup_training_changes",
                "Training changes tested against the selected configuration of \\diana{} on the dev folds, one at a time, same settings, "
                "seeds and planted mislabels. Each cell gives the paired difference, variant minus \\diana{}, in the share of planted "
                "mislabels caught (mean over the 5, 10 and 20\\,\\% false-flag budgets both models can operate at) and in macro-F1. " + RULE,
                [(None, rows(tc, CHANGES, "DIANA"))])
    (TABLES / "sup_table_representations.tex").write_text(rep)
    (TABLES / "sup_table_training_changes.tex").write_text(trn)
    n = tc[tc.b == "DIANA"].rule.value_counts().to_dict()
    print("wrote both tables; DIANA cells (inputs + training changes):", n,
          "| logistic regression cells:", v[v.b == "LogReg"].rule.value_counts().to_dict())


if __name__ == "__main__":
    main()
