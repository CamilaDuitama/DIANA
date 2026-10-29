#!/usr/bin/env python3
"""Supplementary tables of the v19 held-out results (manuscript repository, tables/).

  sup_table_regime_classes.tex   every evaluable class by field and regime (few < 20, medium 20-99,
                                 many >= 100 training runs): training runs and BioProjects, held-out
                                 runs and BioProjects
  sup_table_detection_heldout.tex  planted-mislabel detection on the held-out set, single-label and
                                 whole-record benchmarks: share caught at 5 / 10 / 20 % false flags,
                                 "can't check" where the budget cannot be operated at
  sup_table_nocall.tex           DIANA's no-call cutoffs, answered share, accuracy and calibration error
  sup_table_real_mislabels.tex   the real curator corrections, with each model's flag and suggested label

    ./env/bin/python scripts/paper/generate_sup_tables_results_v19.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diana.evaluation import scoring as S  # noqa: E402

TABLES = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA/tables")
SPLIT = ROOT / "data/splits_v19"
CUTOFFS = ROOT / "results/prior_v19net/abstention_prior_coverage80.json"   # no-call cutoffs, fixed-coverage rule
FIELDS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
NAMES = {"DIANA": "\\diana{}", "LogReg": "Logistic regression", "RandomForest": "Random forest", "kNN": "$k$-NN", "Majority": "Majority class"}
REGIMES = ["many", "medium", "few"]


def tex(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


def regime_classes() -> str:
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    lines = []
    for f, name in FIELDS.items():
        by = S.regime_classes(tr.set_index("Run_accession")[f], S.eligible_classes(SPLIT / "class_eligibility.tsv", f))
        for reg in REGIMES:
            label = reg
            for c in sorted(by[reg], key=lambda c: (-int((tr[f] == c).sum()), c)):
                trc, tec = tr[tr[f] == c], te[te[f] == c]
                held = f"{len(tec)} ({tec.archive_project.nunique()})" if len(tec) else "0"
                cls = f"\\textit{{{tex(c)}}}" if f == "sample_host" else tex(c)
                lines.append(f"{name} & {label} & {cls} & {len(trc)} ({trc.archive_project.nunique()}) & {held} \\\\")
                name, label = "", ""
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    return ("\\centering\n\\caption{Evaluable classes by field and regime. Regimes are set by the number of training runs of a class "
            "(few: fewer than 20; medium: 20 to 99; many: 100 or more). "
            "In columns \"Training\" and \"Held-out\" you can see Runs and, in brackets, BioProjects in the training and held-out sets.}\n"
            "\\label{tab:sup_regime_classes}\n\\scriptsize\n\\begin{tabular}{lllrr}\n\\toprule\n"
            "Field & Regime & Class & Training & Held-out \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def fmt(v, lo=None, hi=None) -> str:
    if pd.isna(v):
        return "can't check"
    return f"{v:.2f}" if lo is None or pd.isna(lo) else f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def detection() -> str:
    lines = []
    for bench, path in (("single label", "flag_single"), ("whole record", "flag_record")):
        d = pd.read_csv(ROOT / f"results/heldout_v19/scores_net/{path}/flagging_scores.tsv", sep="\t").set_index(["task", "model"])
        first = True
        for f, name in FIELDS.items():
            for m in NAMES:
                r = d.loc[(f, m)]
                lines.append(f"{bench if first else ''} & {name if m == 'DIANA' else ''} & {NAMES[m]} & "
                             f"{fmt(r.recall_at_5pct, r.recall_at_5pct_ci_low, r.recall_at_5pct_ci_high)} & {fmt(r.recall_at_10pct)} & {fmt(r.recall_at_20pct)} \\\\")
                first = False
        lines.append("\\midrule")
    return ("\\centering\n\\caption{Planted-mislabel detection on the held-out set: share of planted errors caught when at most 5, 10 or 20\\,\\% "
            "of correctly labelled runs are flagged, mean over 20 plantings; 95\\,\\% interval over held-out BioProjects for the 5\\,\\% budget. "
            "``Can't check'': more correctly labelled runs tie at the highest flag score than the budget allows.}\n"
            "\\label{tab:sup_detection_heldout}\n\\scriptsize\n\\begin{tabular}{lllccc}\n\\toprule\n"
            "Benchmark & Field & Model & 5\\,\\% & 10\\,\\% & 20\\,\\% \\\\\n\\midrule\n" + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def nocall() -> str:
    d = pd.read_csv(ROOT / "results/heldout_v19/diana_net_prior/nocall_calibration_test.tsv", sep="\t").set_index("task")
    lines = [f"{FIELDS[f]} & {'none' if r.cutoff == 0 else f'{r.cutoff:.2f}'} & {r.answered_share * 100:.0f}\\,\\% & {r.accuracy_answered:.2f} & "
             f"{r.accuracy_all:.2f} & {r.ece:.2f} \\\\" for f, r in d.loc[list(FIELDS)].iterrows()]
    return ("\\centering\n\\caption{\\diana's no call and calibration on the held-out set, per field. Cutoff: \\diana answers a run only when its "
            "top probability is at least this value (set on the dev folds, Methods; none for sample host). Answered: share of the scored held-out runs "
            "that \\diana answers. Accuracy, answered: share of the answered runs whose top prediction is correct. Accuracy, all: the same over all "
            "scored runs, without the cutoff. ECE (expected calibration error): mean gap between the top probability and the observed accuracy over ten "
            "probability bins, weighted by runs; 0 means the probabilities match the observed accuracy (Supplementary Figure~\\ref{fig:sup_reliability}).}\n"
            "\\label{tab:sup_nocall}\n\\small\n\\begin{tabular}{lccccc}\n\\toprule\n"
            "Field & Cutoff & Answered & Accuracy, answered & Accuracy, all & ECE \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def real_mislabels() -> str:
    d = pd.read_csv(ROOT / "results/heldout_v19/real_mislabels_net_prior/replay.tsv", sep="\t")
    keys = ["archive_project", "field", "old", "new", "commit"]
    out = []
    for k, g in d.groupby(keys, sort=False):
        n = g.Run_accession.nunique()
        cell = []
        for m in ("DIANA", "LogReg", "RandomForest"):
            s = g[g.model == m]
            flag = "n/a" if (s.verdict == "cannot assess").all() else f"{int(s.flagged_5pct.sum())}"
            cell.append(f"{flag} / {int(s.prediction_is_corrected_label.sum())}")
        out.append(f"{k[0]} & {tex(FIELDS[k[1]].lower())} & {tex(k[2])} $\\rightarrow$ {tex(k[3])} & {n} & \\texttt{{{k[4]}}} & " + " & ".join(cell) + " \\\\")
    return ("\\centering\n\\caption{Real label corrections on held-out runs. Each row is a correction the \\AncientMetagenomeDir{} curators made (old $\\rightarrow$ new label), with the number of runs it affected and the curation commit. For each model: runs where it flags the old label as wrong (at 5\\,\\% false flags) / runs where it predicts the new label. Where the old label is not a training class, it cannot be flagged (n/a).}\n"
            "\\label{tab:sup_real_mislabels}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{lllrlccc}\n\\toprule\n"
            "BioProject & Field & Correction (old $\\rightarrow$ new) & Runs & Commit & \\diana{} & Logistic regression & Random forest \\\\\n\\midrule\n"
            + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def diana_probs(task: str) -> pd.DataFrame:
    p = pd.read_csv(ROOT / f"results/heldout_v19/diana_net/probs_DIANA_{task}.tsv", sep="\t").set_index("Run_accession")
    return p[[c for c in p.columns if c.startswith("p_")]]


def diana_confidence(task: str) -> pd.DataFrame:
    """DIANA's label (argmax of its probabilities) and reported confidence (class prior added back; 85_)."""
    return pd.read_csv(ROOT / f"results/heldout_v19/diana_net_prior/confidence_{task}.tsv", sep="\t",
                       dtype={"Run_accession": str}).set_index("Run_accession")


def perclass() -> str:
    """DIANA (released) per evaluable class on the held-out set: recall, precision and F1 of the top class, no abstention."""
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    lines = []
    for f, name in FIELDS.items():
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", f)
        group = {c: r for r, cs in S.regime_classes(tr.set_index("Run_accession")[f], el).items() for c in cs}
        ok = S.scored_heldout(te[f], set(tr[f].dropna()))
        y = te.loc[ok[ok].index, f]
        pred = diana_probs(f).loc[y.index].idxmax(axis=1).str[2:]
        for c in sorted(el, key=lambda c: (-int((tr[f] == c).sum()), c)):
            trc, hit, npred = tr[tr[f] == c], (y == c), int((pred == c).sum())
            rec = float((pred[hit] == c).mean()) if hit.any() else np.nan
            prec = float((y[pred == c] == c).mean()) if npred else np.nan
            f1 = 0.0 if rec == 0 else (2 * rec * prec / (rec + prec) if np.isfinite(prec) else np.nan)
            cls = f"\\textit{{{tex(c)}}}" if f == "sample_host" else tex(c)
            cells = ["n/a" if not np.isfinite(v) else f"{v:.2f}" for v in (rec, prec, f1)]
            lines.append(f"{name} & {cls} & {group[c]} & {len(trc)} ({trc.archive_project.nunique()}) & "
                         f"{int(hit.sum())} ({te.loc[hit[hit].index, 'archive_project'].nunique()}) & " + " & ".join(cells) + " \\\\")
            name = ""
        lines.append("\\midrule")
    return ("\\centering\n\\caption{\\diana on the held-out set, per evaluable class: training runs and held-out runs (BioProjects in brackets), and the "
            "recall, precision and F1 of \\diana's top prediction (no abstention). Precision is not defined (n/a) for a class \\diana never predicts.}\n"
            "\\label{tab:sup_perclass_heldout}\n\\scriptsize\n\\begin{tabular}{lllrrccc}\n\\toprule\n"
            "Field & Class & Group & Training & Held-out & Recall & Precision & F1 \\\\\n\\midrule\n" + "\n".join(lines[:-1])
            + "\n\\bottomrule\n\\end{tabular}\n")


def unseen_predictions() -> str:
    """DIANA's top prediction for held-out runs whose label is absent from training, and how many fall below the no-call cutoff."""
    import json
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    cut = json.load(open(CUTOFFS))["min_confidence"]
    lines, fields = [], []
    for f, name in FIELDS.items():
        un = te[f].notna() & ~te[f].isin(set(tr[f].dropna()))
        if not un.any():
            continue
        fields.append(name.lower())
        y = te.loc[un[un].index, f]
        L = diana_confidence(f).loc[y.index]
        d = pd.DataFrame({"true": y, "pred": L.label, "nocall": L.confidence < cut[f]})
        it = (lambda x: f"\\textit{{{tex(x)}}}") if f == "sample_host" else tex
        for (t, p), g in d.groupby(["true", "pred"]):
            lines.append(f"{name} & {it(t)} & {it(p)} & {len(g)} & {int(g.nocall.sum())} \\\\")
            name = ""
    return ("\\centering\n\\caption{\\diana's top prediction for held-out runs whose label is absent from training (" + ", ".join(fields)
            + " only; no other field has such runs). No call: runs whose confidence is below the field's no-call cutoff (Methods).}\n"
            "\\label{tab:unseen_labels}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lllrr}\n\\toprule\n"
            "Field & Label (absent from training) & \\diana's prediction & Runs & No call \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def consensus(min_runs: int = 5) -> str:
    """Labels contradicted by DIANA and the three baselines together (results/consensus_v19net, 76_), groups of at least `min_runs` runs."""
    import re
    d = pd.read_csv(ROOT / "results/consensus_v19net/summary.tsv", sep="\t")
    total = int(d.runs.sum())
    g = d.groupby(["field", "curated", "consensus", "project_name"], as_index=False).runs.sum()
    g = g[g.runs >= min_runs].sort_values(["runs", "field"], ascending=[False, True])
    study = lambda p: re.sub(r"([A-Za-z])(\d{4})$", r"\1 \2", p)
    it = lambda f, x: f"\\textit{{{tex(x)}}}" if f == "sample_host" else tex(x)
    lines = [f"{FIELDS[r.field]} & {tex(study(r.project_name))} & {it(r.field, r.curated)} & {it(r.field, r.consensus)} & {r.runs} \\\\"
             for r in g.itertuples()]
    return ("\\centering\n\\caption{Curated labels that \\diana, logistic regression, random forest and $k$-NN all contradict with the same "
            "prediction, each model predicting runs it was not trained on (out-of-fold for training runs). Groups of at least "
            f"{min_runs} runs by field and publication, named as in AncientMetagenomeDir ({int(g.runs.sum())} of the {total} run labels contradicted this way).}}\n"
            "\\label{tab:sup_consensus}\n\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{lllll}\n\\toprule\n"
            "Field & Publication & Curated label & All models predict & Runs \\\\\n\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def depth_control() -> str:
    """Read count only against unitig fractions (results/depth_control_tuned_v19net, 81_): held-out and dev-fold macro-F1 of
    DIANA and of logistic regression and random forest on each input, all tuned on the dev folds. The caption states that
    every read-count model is below the same model on fractions and below DIANA; this is checked here, and comparisons whose
    interval includes zero are named in the caption (macro-F1 must have none)."""
    D = ROOT / "results/depth_control_tuned_v19net"
    sc = pd.read_csv(D / "scores.tsv", sep="\t").set_index(["task", "split", "metric", "model"]).value
    pr = pd.read_csv(D / "paired.tsv", sep="\t")
    if (pr.delta <= 0).any():
        raise ValueError("a read-count model is not below its comparison; the caption no longer holds")
    open_ = pr[(pr.ci_low <= 0) & (pr.ci_high >= 0)]
    if (open_.metric == "macro_f1").any():
        raise ValueError("a macro-F1 interval includes zero; the caption no longer holds")
    short = {"DIANA": "\\diana{}", "LogReg": "LR on unitig fractions", "RandomForest": "RF on unitig fractions",
             "Reads_LogReg": "LR on read count", "Reads_RandomForest": "RF on read count"}
    exceptions = "; ".join(f"{FIELDS[r.task].lower()}, {r.split}, {short[r.a]} against {short[r.b]}: ${r.delta:+.2f}$, interval "
                           f"${r.ci_low:.2f}$ to ${r.ci_high:.2f}$" for r in open_.itertuples())
    lines = []
    for t, name in FIELDS.items():
        for split in ("dev folds", "held-out"):
            cells = [f"{sc[(t, split, 'macro_f1', m)]:.2f}" for m in ("DIANA", "LogReg", "Reads_LogReg", "RandomForest", "Reads_RandomForest")]
            lines.append(f"{name if split == 'dev folds' else ''} & {split} & " + " & ".join(cells) + " \\\\")
    return ("\\centering\n\\caption{Sequencing-depth control, macro-F1 on the dev folds (out-of-fold) and the held-out set. Logistic "
            "regression (LR) and random forest (RF) were given either the unitig fractions or only the run's read count, each tuned on "
            "the dev folds. Every read-count model is below the same model on unitig fractions and below \\diana, with a 95\\,\\% "
            "interval over BioProjects excluding zero, on both sets"
            + (f"; in balanced accuracy all intervals but one exclude zero ({exceptions})" if len(open_) == 1 else
               f"; in balanced accuracy all intervals but {len(open_)} exclude zero ({exceptions})" if len(open_) else
               ", and also for balanced accuracy") + ".}\n"
            "\\label{tab:sup_depth}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llccccc}\n\\toprule\n"
            " & & & \\multicolumn{2}{c}{LR} & \\multicolumn{2}{c}{RF} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Field & Set & \\diana{} & Unitig fractions & Read count & Unitig fractions & Read count \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def labelling_aid() -> str:
    """DIANA as a labelling aid on the held-out set: per field and class group, scored runs, share answered (no-call cutoffs)
    and accuracy on the answered runs and on all runs."""
    import json
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    cut = json.load(open(CUTOFFS))["min_confidence"]
    lines = []
    for f, name in FIELDS.items():
        el = S.eligible_classes(SPLIT / "class_eligibility.tsv", f)
        grp = {c: g for g, cs in S.regime_classes(tr[f], el).items() for c in cs}
        sc = S.scored_heldout(te[f], set(tr[f].dropna())); runs = sc[sc].index
        L = diana_confidence(f).loc[runs]
        pred, y = L.label.to_numpy(), te.loc[runs, f].astype(str).to_numpy()
        g, ans = np.array([grp.get(c, "") for c in y]), L.confidence.to_numpy() >= cut[f]
        for G in REGIMES:
            m = g == G
            if not m.any():
                continue
            acc_a = f"{(pred[m & ans] == y[m & ans]).mean():.2f}" if (m & ans).any() else "n/a"
            lines.append(f"{name} & {G} & {int(m.sum())} & {ans[m].mean() * 100:.0f}\\,\\% & {acc_a} & {(pred[m] == y[m]).mean():.2f} \\\\")
            name = ""
        lines.append("\\midrule")
    return ("\\centering\n\\caption{\\diana as a labelling aid on the held-out set, by field and group of classes (many: at least 100 training "
            "runs; medium: 20 to 99; few: fewer than 20): scored runs, share that \\diana answers (no-call cutoffs, Methods), and accuracy on "
            "the answered runs and on all runs.}\n\\label{tab:sup_labelling_aid}\n\\small\n\\begin{tabular}{llrccc}\n\\toprule\n"
            "Field & Group & Runs & Answered & Accuracy, answered & Accuracy, all \\\\\n\\midrule\n" + "\n".join(lines[:-1])
            + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    for name, text in (("sup_table_regime_classes.tex", regime_classes()), ("sup_table_detection_heldout.tex", detection()),
                       ("sup_table_real_mislabels.tex", real_mislabels()),
                       ("sup_table_perclass_heldout.tex", perclass()), ("sup_table_unseen_predictions.tex", unseen_predictions()), ("sup_table_consensus.tex", consensus()), ("sup_table_depth.tex", depth_control()), ("sup_table_labelling_aid.tex", labelling_aid())):
        (TABLES / name).write_text(text)
        print("wrote", TABLES / name)


if __name__ == "__main__":
    main()


def scored_counts() -> str:
    """Held-out runs and BioProjects scored per field (true class among the training labels), as LaTeX text."""
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    parts = []
    for f, name in FIELDS.items():
        ok = S.scored_heldout(te[f], set(tr[f].dropna()))
        parts.append(f"{name.lower()} {int(ok.sum())} runs from {te.loc[ok[ok].index, 'archive_project'].nunique()} BioProjects")
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def bold_best(vals: dict) -> dict:
    """LaTeX cells, the highest value of the column in bold (ties all bold); None -> em dash."""
    finite = [v for v in vals.values() if v is not None and np.isfinite(v)]
    top = max(finite) if finite else None
    return {m: ("---" if v is None or not np.isfinite(v) else (f"\\textbf{{{v:.2f}}}" if np.isclose(v, top) else f"{v:.2f}"))
            for m, v in vals.items()}


def diana_table() -> str:
    """Main table: the released DIANA on the held-out set per field (classification, no call, calibration, detection)."""
    c = pd.read_csv(ROOT / "results/multiplicity_v19/net_classification_heldout/scores.tsv", sep="\t")
    c = c[(c.regime == "all") & (c.model == "DIANA")].set_index(["task", "metric"])
    n = pd.read_csv(ROOT / "results/heldout_v19/diana_net_prior/nocall_calibration_test.tsv", sep="\t").set_index("task")
    f = pd.read_csv(ROOT / "results/heldout_v19/scores_net/flag_single/flagging_scores.tsv", sep="\t").set_index(["task", "model"])
    tr = pd.read_csv(SPLIT / "train_metadata.tsv", sep="\t", dtype=str).replace("", np.nan)
    te = pd.read_csv(SPLIT / "test_metadata.tsv", sep="\t", dtype=str).replace("", np.nan).set_index("Run_accession")
    cell = lambda t, m: fmt(*c.loc[(t, m), ["value", "ci_low", "ci_high"]])
    runs = {}
    for t in FIELDS:
        ok = S.scored_heldout(te[t], set(tr[t].dropna()))
        if int(ok.sum()) != int(n.at[t, "scored_runs"]):
            raise ValueError(f"{t}: scored runs differ between the split and the no-call table")
        runs[t] = f"{int(ok.sum())} ({te.loc[ok[ok].index, 'archive_project'].nunique()})"
    rows = [("Runs scored (BioProjects)", lambda t: runs[t]),
            ("Balanced accuracy", lambda t: cell(t, "balanced_accuracy")),
            ("Macro-F1", lambda t: cell(t, "macro_f1")),
            ("No-call cutoff", lambda t: "none" if n.at[t, "cutoff"] == 0 else f"{n.at[t, 'cutoff']:.2f}"),
            ("Runs answered", lambda t: f"{n.at[t, 'answered_share'] * 100:.0f}\\,\\%"),
            ("Accuracy on answered runs", lambda t: f"{n.at[t, 'accuracy_answered']:.2f}"),
            ("Calibration error (ECE)", lambda t: f"{n.at[t, 'ece']:.2f}"),
            ("Planted errors caught at 5\\,\\%", lambda t: fmt(*f.loc[(t, "DIANA"), ["recall_at_5pct", "recall_at_5pct_ci_low", "recall_at_5pct_ci_high"]]))]
    body = [f"{lab} & " + " & ".join(fn(t) for t in FIELDS) + " \\\\" for lab, fn in rows]
    body.insert(3, "\\midrule"); body.insert(7, "\\midrule"); body.insert(9, "\\midrule")
    return ("\\centering\n\\caption{\\diana on the held-out set (BioProjects never seen in training), per field. Balanced accuracy and "
            "macro-F1 over the runs whose class occurs in training, without the no-call cutoff; per class: Supplementary Table~\\ref{tab:sup_perclass_heldout}. "
            "No call: \\diana answers a run only when its confidence (its probability for the predicted label, with the class prior added back; "
            "Methods) reaches the cutoff. ECE: mean gap between the confidence and the observed accuracy, 0 when they match (Supplementary Figure~\\ref{fig:sup_reliability}). Planted errors: share caught while flagging at most "
            "5\\,\\% of correctly labelled runs (mean over 20 plantings). Brackets: 95\\,\\% intervals over held-out BioProjects.}\n"
            "\\label{tab:diana_heldout}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lcccc}\n\\toprule\n & "
            + " & ".join(FIELDS.values()) + " \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")


def main_tables() -> tuple[str, str, str]:
    """Table 1 (classification), Table 2 (detection) for the main text; the classification intervals for the supplementary."""
    c = pd.read_csv(ROOT / "results/multiplicity_v19/net_classification_heldout/scores.tsv", sep="\t")
    c = c[c.regime == "all"].set_index(["task", "metric", "model"])
    f = pd.read_csv(ROOT / "results/heldout_v19/scores_net/flag_single/flagging_scores.tsv", sep="\t").set_index(["task", "model"])
    cols = [(t, met) for t in FIELDS for met in ("macro_f1", "balanced_accuracy")]
    cells = {k: bold_best({m: float(c.at[(k[0], k[1], m), "value"]) for m in NAMES}) for k in cols}
    t1 = [f"{NAMES[m]} & " + " & ".join(cells[k][m] for k in cols) + " \\\\" for m in NAMES]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{FIELDS[t]}}}" for t in FIELDS)
    rule = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4))
    n = scored_counts()
    table1 = ("\\centering\n\\caption{Held-out classification on BioProjects never seen in training: macro-F1 (F1) and balanced "
              f"accuracy (BA) over the evaluable classes. Runs scored (those whose class occurs in training): {n}." " Bold: highest value in each column. Intervals: Supplementary Table~\\ref{tab:classification_heldout}; "
              "paired comparisons by group of classes: Figure~\\ref{fig:results_classification}.}\n\\label{tab:performance}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
              "\\begin{tabular}{l" + "cc" * 4 + "}\n\\toprule\n & " + head + " \\\\\n" + rule + "\n Model" + " & F1 & BA" * 4 + " \\\\\n\\midrule\n"
              + "\n".join(t1) + "\n\\bottomrule\n\\end{tabular}\n")
    det = {t: bold_best({m: (None if pd.isna(f.at[(t, m), "recall_at_5pct"]) else float(f.at[(t, m), "recall_at_5pct"])) for m in NAMES}) for t in FIELDS}
    t2 = [f"{NAMES[m]} & " + " & ".join(det[t][m] for t in FIELDS) + " \\\\" for m in NAMES if m not in ("kNN", "Majority")]
    if any(det[t][m] != "---" for t in FIELDS for m in ("kNN", "Majority")):
        raise ValueError("k-NN or the majority class can operate at 5 %; they cannot be left out of Table 2")
    table2 = ("\\centering\n\\caption{Held-out mislabel detection: share of planted label errors flagged by each model when it may flag at most 5\\,\\% "
              "of the correctly labelled runs (mean over 20 plantings). Bold: highest value in each column. $k$-NN and the majority class are not shown: "
              "they give their highest flag score to more than 5\\,\\% of the correctly labelled runs, so they cannot be used at this budget. Intervals and the 10 and 20\\,\\% "
              "budgets: Supplementary Table~\\ref{tab:sup_detection_heldout}; paired comparisons by group of classes: Figure~\\ref{fig:results_detection}.}\n"
              "\\label{tab:detection}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\nModel & " + " & ".join(FIELDS.values()) + " \\\\\n\\midrule\n"
              + "\n".join(t2) + "\n\\bottomrule\n\\end{tabular}\n")
    sup = []
    for t, name in FIELDS.items():
        for m in NAMES:
            vals = [fmt(*c.loc[(t, met, m), ["value", "ci_low", "ci_high"]]) for met in ("macro_f1", "balanced_accuracy")]
            sup.append(f"{name if m == 'DIANA' else ''} & {NAMES[m]} & " + " & ".join(vals) + " \\\\")
        sup.append("\\midrule")
    table_s = ("\\centering\n\\caption{Held-out classification with 95\\,\\% intervals over held-out BioProjects (2,000 resamples): macro-F1 and balanced "
               "accuracy over the evaluable classes.}\n\\label{tab:classification_heldout}\n\\small\n\\begin{tabular}{llcc}\n\\toprule\n"
               "Field & Model & Macro-F1 & Balanced accuracy \\\\\n\\midrule\n" + "\n".join(sup[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    return table1, table2, table_s


if __name__ == "__main__" and "--main-tables" in sys.argv:
    t1, t2, ts = main_tables()
    for name, text in (("main_table_00_diana_heldout.tex", diana_table()), ("main_table_01_performance_summary.tex", t1),
                       ("main_table_02_detection.tex", t2), ("sup_table_classification_heldout.tex", ts)):
        (TABLES / name).write_text(text)
        print("wrote", TABLES / name)
