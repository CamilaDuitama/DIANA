#!/usr/bin/env python3
"""Main tables and figures of the revised manuscript, from the final held-out evaluation
(read 11) and the dev-fold artefacts. DIANA = the four per-task networks averaged fifty-fifty
with the tuned logistic regression (decision of 2026-10-05); every number here is that model's
unless a column names a baseline.

Writes LaTeX tables and PNG/PDF figures to results/paper_v2/ and, when --manuscript is given,
copies them into the manuscript repository's tables/ and figures/ directories (uncommitted).

    ./env/bin/python scripts/paper/generate_manuscript_v2_main_assets.py \\
        --manuscript /pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
F = __import__("59_devfold_flagging_score")
from diana.evaluation.metrics import load_eligible_classes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v2"
TASKS = ["community_type", "feature", "sample_host", "material"]
PRETTY = {"community_type": "Community Type", "feature": "Feature", "sample_host": "Sample Host", "material": "Material"}
TE = pd.read_csv(ROOT / "data/splits_v9/test_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")
TR = pd.read_csv(ROOT / "data/splits_v9/train_metadata.tsv", sep="\t", low_memory=False).set_index("Run_accession")


def heldout_net(task):
    src = ROOT / "results/final_test_v9" / ("studyweight" if task == "feature" else "reference")
    dirs = [src / task / f"seed{s}" for s in (42, 1, 2, 3, 4)]
    runs = [l.strip() for l in open(dirs[0] / "eval_runs.txt") if l.strip()]
    classes = json.load(open(dirs[0] / "label_classes.json"))[task]
    return pd.DataFrame(np.mean([np.load(d / "probs_last.npz")[task] for d in dirs], axis=0), index=runs, columns=classes)


def heldout_lr(task):
    d = pd.read_csv(ROOT / f"results/baseline_predictions_v9/heldout_probabilities_{task}.tsv", sep="\t")
    d = d[d.model == "LogisticRegression_Bal"].set_index("Run_accession")
    return d[[c for c in d.columns if c.startswith("p_")]].rename(columns=lambda c: c[2:])


def avg(a, b):
    cols = sorted(set(a.columns) | set(b.columns)); idx = a.index.intersection(b.index)
    return (a.reindex(index=idx, columns=cols).fillna(0.0) + b.reindex(index=idx, columns=cols).fillna(0.0)) / 2


def diana_heldout(task):
    return avg(heldout_net(task), heldout_lr(task))


def oof_net(task):
    return F.reference_probs(task, ROOT / ("results/step4_2_studyweight_v9" if task == "feature" else "results/reference_v9"))


def diana_oof(task):
    return avg(oof_net(task), F.baseline_probs("LogisticRegression_Bal", task))


# ---------------------------------------------------------------- table 1: classification + detection
def table1(manuscript: Path | None):
    C = pd.read_csv(ROOT / "results/final_test_v9/scores/classification.tsv", sep="\t")
    FL = pd.read_csv(ROOT / "results/final_test_v9/scores/flagging.tsv", sep="\t")
    FL = FL[FL.bench == "single"]
    name = {"DIANA+LogReg average": "\\diana", "LogisticRegression_Bal": "Logistic regression", "RandomForest_Bal": "Random forest",
            "kNN": "k-NN", "LinearSVM_Bal": "Linear SVM", "MajorityClass": "Majority class"}
    order = ["DIANA+LogReg average", "LogisticRegression_Bal", "RandomForest_Bal", "kNN", "LinearSVM_Bal", "MajorityClass"]
    def cells(t, m):
        c = C[(C.task == t) & (C.model == m)]
        f = FL[(FL.task == t) & (FL.model == m)]
        cls = f"{c.f1_macro_eligible.iloc[0]:.2f} [{c.ci_low.iloc[0]:.2f}, {c.ci_high.iloc[0]:.2f}]" if len(c) else "--"
        if len(f) and not bool(f.cant_check_5pct.iloc[0]):
            det = f"{f.recall_at_5pct.iloc[0]:.2f} [{f.ci_low_5pct.iloc[0]:.2f}, {f.ci_high_5pct.iloc[0]:.2f}]"
        elif len(f):
            det = "cannot operate"
        else:
            det = "--"
        return cls, det

    # main table: detection only (Camila 2026-10-06); the linear SVM has no
    # probabilities, so it appears only in the supplementary classification table.
    det_order = [m for m in order if m != "LinearSVM_Bal"]
    rows = []
    for t in TASKS:
        for i, m in enumerate(det_order):
            _, det = cells(t, m)
            lead = f"{PRETTY[t]}" if i == 0 else ""
            rows.append(f"{lead} & {name[m]} & {det} \\\\")
        rows.append("\\addlinespace")
    tex = """\\centering
\\caption{Held-out mislabel detection by \\diana and the baselines (922 runs from 34 BioProjects never seen in training): the share of planted label errors caught while flagging at most 5\\,\\% of correctly labelled runs (single-label benchmark, mean over 20 seeded plantings). Brackets are 95\\,\\% intervals from resampling BioProjects. ``Cannot operate'' marks models whose tied top scores exceed the 5\\,\\% budget. Classification scores for all models, including the linear SVM, are given in Supplementary Table~\\ref{tab:classification_heldout}.}
\\label{tab:performance}
\\small
\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llc@{}}
\\toprule
Task & Model & Recall at 5\\,\\% false flags \\\\
\\midrule
""" + "\n".join(rows[:-1]) + """
\\bottomrule
\\end{tabular*}
\\par\\vspace{4pt}
\\parbox{\\linewidth}{\\footnotesize \\textbf{Notes:} \\diana{} = the four per-task networks (five seeds averaged) combined half-and-half with the tuned logistic regression. Paired study-level tests resolve two cells in \\diana's favour (Feature against its network component and Material against logistic regression); every other pairwise comparison is a tie. Denominators and out-of-vocabulary counts per task are given in the supplement.}
"""
    (OUT / "main_table_01_performance_summary.tex").write_text(tex)

    # supplementary table: classification for all six models
    rows = []
    for t in TASKS:
        for i, m in enumerate(order):
            cls, _ = cells(t, m)
            lead = f"{PRETTY[t]}" if i == 0 else ""
            rows.append(f"{lead} & {name[m]} & {cls} \\\\")
        rows.append("\\addlinespace")
    sup = """\\centering
\\caption{Held-out classification of \\diana and the baselines (922 runs from 34 BioProjects never seen in training): macro-F1 over the classes observable out of study. Brackets are 95\\,\\% intervals from resampling BioProjects.}
\\label{tab:classification_heldout}
\\small
\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llc@{}}
\\toprule
Task & Model & Macro-F1 (eligible classes) \\\\
\\midrule
""" + "\n".join(rows[:-1]) + """
\\bottomrule
\\end{tabular*}
"""
    (OUT / "sup_table_classification_heldout.tex").write_text(sup)
    if manuscript:
        shutil.copy(OUT / "main_table_01_performance_summary.tex", manuscript / "tables/main_table_01_performance_summary.tex")
        shutil.copy(OUT / "sup_table_classification_heldout.tex", manuscript / "tables/sup_table_classification_heldout.tex")


# ---------------------------------------------------------------- confusion matrices (held-out, DIANA)
def confusions(manuscript: Path | None):
    for t in TASKS:
        P = diana_heldout(t)
        lab = TE[t].reindex(P.index); k = lab.notna()
        y = lab[k].astype(str); p = P[k.to_numpy()].idxmax(axis=1).astype(str)
        elig = load_eligible_classes(ROOT / "data/splits_v9/class_eligibility.tsv", t)
        classes = sorted(set(y) & elig)
        yk = y.isin(classes) & p.notna()
        cm = confusion_matrix(y[yk], p[yk].where(p[yk].isin(classes), other="other"), labels=classes + ["other"])
        cm = cm[:len(classes), :]
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(classes) + 2), max(4, 0.5 * len(classes) + 1.5)))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes) + 1)); ax.set_xticklabels(classes + ["other"], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes) + 1):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel(f"label on file ({PRETTY[t]})")
        ax.set_title(f"Held-out confusion matrix, {PRETTY[t]}")
        fig.colorbar(im, ax=ax, fraction=0.035, label="row share")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"main_01_confusion_matrix_{t}.{ext}", dpi=200)
        plt.close(fig)
        if manuscript:
            shutil.copy(OUT / f"main_01_confusion_matrix_{t}.png", manuscript / f"figures/main_01_confusion_matrix_{t}.png")


# ---------------------------------------------------------------- operating curves (held-out)
def operating_curves(manuscript: Path | None):
    plants = [pd.read_csv(f, sep="\t").set_index("Run_accession")
              for f in sorted(glob.glob(str(ROOT / "results/planted_mislabels_heldout_v9/planted_test_mixed_r0.1_s*.tsv")))]
    budgets = np.linspace(0.0, 0.3, 61)
    for t in TASKS:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for mname, P, color in (("DIANA", diana_heldout(t), "#2a78d6"), ("Logistic regression", heldout_lr(t), "#eb6834"),
                                ("Random forest", None, "#8a877d")):
            if mname == "Random forest":
                d = pd.read_csv(ROOT / f"results/baseline_predictions_v9/heldout_probabilities_{t}.tsv", sep="\t")
                d = d[d.model == "RandomForest_Bal"].set_index("Run_accession")
                P = d[[c for c in d.columns if c.startswith("p_")]].rename(columns=lambda c: c[2:])
            curves = []
            for pl in plants:
                had = TE.loc[pl.index, t].notna().to_numpy()
                pl_t = pl[pl[t].notna() & pl[f"{t}_planted"].notna() & had]
                sc = F.flag_scores(P, pl_t[t]).dropna()
                planted = pl_t.loc[sc.index, f"{t}_planted"].astype(bool).to_numpy()
                curves.append([F.recall_at_budget(sc.to_numpy(), planted, b)[0] if 0 < b else 0.0 for b in budgets])
            m = np.nanmean(curves, axis=0)
            ax.plot(budgets, m, label=mname, color=color, lw=1.8)
        ax.axvline(0.05, color="#bbb", lw=1, ls="--")
        ax.text(0.052, 0.03, "5 % budget", fontsize=7, color="#777")
        ax.set_xlabel("false-flag budget (share of correct runs flagged)")
        ax.set_ylabel("planted errors caught")
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 0.3)
        ax.set_title(f"Operating curve, {PRETTY[t]}")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"main_02_operating_curve_{t}.{ext}", dpi=200)
        plt.close(fig)
        if manuscript:
            shutil.copy(OUT / f"main_02_operating_curve_{t}.png", manuscript / f"figures/main_02_operating_curve_{t}.png")


# ---------------------------------------------------------------- support law (held-out, per class)
def support_law(manuscript: Path | None):
    rows = []
    for t in TASKS:
        P = diana_heldout(t)
        lab = TE[t].reindex(P.index); k = lab.notna()
        y = lab[k].astype(str); p = P[k.to_numpy()].idxmax(axis=1).astype(str)
        elig = load_eligible_classes(ROOT / "data/splits_v9/class_eligibility.tsv", t)
        nproj = TR.groupby(t).archive_project.nunique()
        for c in sorted(set(y) & elig):
            rows.append({"task": t, "class": c, "train_studies": int(nproj.get(c, 0)),
                         "F1": f1_score(y, p, labels=[c], average="macro", zero_division=0)})
    d = pd.DataFrame(rows); d = d[d.train_studies > 0]
    colors = {"community_type": "#2a78d6", "feature": "#eb6834", "sample_host": "#1baf7a", "material": "#eda100"}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for t in TASKS:
        g = d[d.task == t]
        ax.scatter(g.train_studies, g.F1, s=34, color=colors[t], label=PRETTY[t], edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(6, color="#bbb", lw=1, ls="--", zorder=1)
    ax.text(6.3, 0.02, "6 studies", fontsize=7, color="#777")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 25]); ax.set_xticklabels([1, 2, 5, 10, 25])
    ax.set_xlabel("training BioProjects carrying the class")
    ax.set_ylabel("held-out F1 of the class")
    ax.set_title("Per-class F1 by training support (held-out)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"main_03_class_support_f1.{ext}", dpi=200)
    plt.close(fig)
    d.to_csv(OUT / "main_03_class_support_f1.tsv", sep="\t", index=False)
    if manuscript:
        shutil.copy(OUT / "main_03_class_support_f1.png", manuscript / "figures/main_03_class_support_f1.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manuscript", type=Path, default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    table1(a.manuscript); print("table 1 written")
    confusions(a.manuscript); print("confusion matrices written")
    operating_curves(a.manuscript); print("operating curves written")
    support_law(a.manuscript); print("support figure written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
