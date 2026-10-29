#!/usr/bin/env python3
"""Supplementary tables of the dev-fold screens on the current split (PROTOCOLS.md, v19 amendments of 2026-10-21/22).

From results/screens_v19/verdicts_net.tsv (72_screen_verdicts_v19.py; DIANA = the shared network alone, PROTOCOLS.md v19
amendment 2026-10-26) and, for equal weight per study and for averaging with logistic regression, the re-scored dev-fold
tables (results/rescore_v19/dev_single_{studyweight_net,composition}/, results/classification_v19/dev_{studyweight_net,composition}/),
classified with the same rule (72_ verdict). Each cell: paired difference in the share of planted mislabels
caught (mean over the budgets both models can operate at) / paired difference in macro-F1, variant minus
reference; ^- marks a loss and ^+ a win under the rule. Writes, in the manuscript repository:
  tables/sup_table_representations.tex  input representations (DIANA, then logistic regression)
  tables/sup_table_training_changes.tex training changes (DIANA)
and the same results as grids (plotly, colours from config.py) in results/paper_v19/:
  sup_screens_training_changes.png, sup_screens_representations.png
(cell colour: the side the rule favours; text: detection difference, then macro-F1 difference).

    ./env/bin/python scripts/paper/generate_sup_tables_screens_v19.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import PLOT_CONFIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/paper_v19"
C, V = PLOT_CONFIG["colors"]["models"], PLOT_CONFIG["colors"]["palette"]
FIELDS = {"community_type": "Community type", "feature": "Feature", "sample_host": "Sample host", "material": "Material"}
TIE = "#e6e6e6"
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
    "adv_conf0.1": "BioProject-adversarial training, strength 0.1", "adv_conf0.3": "BioProject-adversarial training, strength 0.3",
    "adv_conf1.0": "BioProject-adversarial training, strength 1.0",
    "lr_average": "averaged with logistic regression",
    "temperature": "temperature scaling", "confident_learning": "Confident Learning flag score",
    "prior": "class frequencies added back",
}
# Changes to the probabilities after training (labels unchanged, so macro-F1 difference 0): dev-fold detection, DIANA variant minus DIANA.
POST = {"temperature": ("results/rescore_v19/dev_single_calibration_net/flagging_paired.tsv", "DIANAcal"),
        "confident_learning": ("results/flagging_v19/dev_cl_net_mixed/flagging_paired.tsv", "DIANA_CL"),
        "prior": ("results/prior_v19net/flag_mixed/flagging_paired.tsv", "DIANAprior")}
# Changes re-scored from stored dev-fold outputs: (detection table, classification table, variant, reference).
PAIRED = {"studyweight": ("results/rescore_v19/dev_single_studyweight_net/flagging_paired.tsv",
                          "results/classification_v19/dev_studyweight_net/classification_paired.tsv", "DIANAW", "DIANA"),
          "lr_average": ("results/rescore_v19/dev_single_composition/flagging_paired.tsv",
                         "results/classification_v19/dev_composition/classification_paired.tsv", "SharedLR", "Shared")}
RULE = ("A variant wins a field ($^{+}$) only if it catches more planted mislabels (paired 95\\,\\% interval over BioProjects above zero, "
        "2,000 draws) and its macro-F1 does not clearly drop; it loses ($^{-}$) if it catches fewer, or if its macro-F1 interval lies entirely "
        "below zero with a drop of at least the field's seed-noise threshold (0.080, 0.042, 0.028 and 0.017 for community type, feature, "
        "sample host and material); all other cells are ties.")


def verdict_rule():
    spec = importlib.util.spec_from_file_location("v72", ROOT / "scripts/analysis/72_screen_verdicts_v19.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verdict


def paired_rows() -> pd.DataFrame:
    out = []
    for arm, (flag, cls, a, b) in PAIRED.items():
        f, c = pd.read_csv(ROOT / flag, sep="\t"), pd.read_csv(ROOT / cls, sep="\t")
        m = c.merge(f, on=["task", "a", "b"], suffixes=("_f1", "_flag"), validate="one_to_one")
        m = m[(m.a == a) & (m.b == b)]
        if len(m) != len(TASKS):
            raise ValueError(f"{arm}: {len(m)} fields for {a} vs {b}")
        out.append(pd.DataFrame({"arm": arm, "task": m.task, "b": "DIANA", "flag_delta": m.delta_choosing,
                                 "flag_low": m.ci_low_flag, "flag_high": m.ci_high_flag, "flag_verdict": m.verdict_flag,
                                 "f1_delta": m.delta, "f1_low": m.ci_low_f1, "f1_high": m.ci_high_f1}))
    d = pd.concat(out, ignore_index=True)
    d["rule"] = d.apply(verdict_rule(), axis=1)
    return d


def post_rows() -> pd.DataFrame:
    out = []
    for arm, (path, a) in POST.items():
        f = pd.read_csv(ROOT / path, sep="\t")
        f = f[(f.a == a) & (f.b == "DIANA")]
        out.append(pd.DataFrame({"arm": arm, "task": f.task, "b": "DIANA", "flag_delta": f.delta_choosing, "flag_low": f.ci_low,
                                 "flag_high": f.ci_high, "flag_verdict": f.verdict, "f1_delta": 0.0, "f1_low": 0.0, "f1_high": 0.0}))
    d = pd.concat(out, ignore_index=True)
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


def num_grid(x: float, decisive: bool) -> str:
    """Two decimals; three in a coloured cell whose non-zero difference would print as 0.00."""
    return f"{x:+.3f}" if decisive and num(x) == "0.00" and x != 0 else num(x)


def plain(label: str) -> str:
    return label.replace("$k$", "k")


def screen_grid(v: pd.DataFrame, blocks: list[tuple[str, dict, str]], legend: dict, out: Path) -> None:
    """Rows: variants, grouped by reference model; columns: fields. Colour: the reference model's colour when the rule calls a loss,
    the variant colour for a win, grey for a tie. Text: detection difference (bold) and macro-F1 difference, variant minus reference."""
    labels, z, text = [], [], []
    keys = ["tie", "variant"] + [b for _, _, b in blocks]
    for title, arms, b in blocks:
        if len(blocks) > 1:
            labels.append(f"<b>{title}</b>"); z.append([None] * 4); text.append([""] * 4)
        for arm, label in arms.items():
            d = v[(v.arm == arm) & (v.b == b)].set_index("task")
            if set(d.index) != set(FIELDS):
                raise ValueError(f"{arm} vs {b}: fields {sorted(d.index)}")
            labels.append(plain(label))
            z.append([keys.index({"win": "variant", "loss": b}.get(d.at[t, "rule"], "tie")) for t in FIELDS])
            text.append([f"<b>{num_grid(d.at[t, 'flag_delta'], d.at[t, 'rule'] != 'tie')}</b><br><span style='font-size:11px'>F1 {num_grid(d.at[t, 'f1_delta'], d.at[t, 'rule'] != 'tie')}</span>" for t in FIELDS])
    colours = [TIE, V[2]] + [C[b] for b in keys[2:]]
    scale = [[k / len(colours), c] for i, c in enumerate(colours) for k in (i, i + 1)]
    xs, ys = list(range(4)), list(range(len(labels)))
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, colorscale=scale, zmin=-0.5, zmax=len(colours) - 0.5, showscale=False, xgap=3, ygap=3,
                               hoverinfo="skip"))
    fig.add_scatter(x=[j for _ in ys for j in xs], y=[i for i in ys for _ in xs], mode="text", text=[t for r in text for t in r],
                    textfont=dict(size=13, color="#222222"), showlegend=False, hoverinfo="skip")
    for key, name in legend.items():
        fig.add_scatter(x=[None], y=[None], mode="markers", name=name,
                        marker=dict(symbol="square", size=14, color={"tie": TIE, "variant": V[2]}.get(key, C.get(key))))
    fig.update_layout(template=PLOT_CONFIG["template"], width=1000, height=150 + 46 * len(labels), font=dict(size=14, color="#333333"),
                      xaxis=dict(side="top", showgrid=False, zeroline=False, tickvals=xs, ticktext=list(FIELDS.values()), range=[-0.5, 3.5]),
                      yaxis=dict(showgrid=False, zeroline=False, tickvals=ys, ticktext=labels, range=[len(labels) - 0.5, -0.5]),
                      legend=dict(orientation="h", yanchor="top", y=-0.01, xanchor="left", x=0), margin=dict(l=360, r=20, t=60, b=80))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.write_image(out, scale=2)


def main() -> None:
    v = pd.read_csv(ROOT / "results/screens_v19/verdicts_net.tsv", sep="\t")
    lr = pd.read_csv(ROOT / "results/screens_v19/verdicts.tsv", sep="\t")   # logistic regression pairs do not involve DIANA
    v = pd.concat([v[v.b == "DIANA"], lr[lr.b == "LogReg"]], ignore_index=True)
    rep = table("tab:sup_representations",
                "Input representations tested against the unitig fraction on the dev folds. Each cell gives the paired difference, "
                "alternative minus fraction, in the share of planted mislabels caught (mean over the 5, 10 and 20\\,\\% false-flag budgets "
                "both models can operate at) and in macro-F1. Upper block: \\diana{} with only its input changed; lower block: logistic "
                "regression alone. Projections and scalings were fitted on the "
                "training folds of each dev fold only. " + RULE,
                [("\\diana{}", rows(v, INPUTS, "DIANA")), ("logistic regression", rows(v, {k: x for k, x in INPUTS.items()}, "LogReg"))])
    tc = pd.concat([v, paired_rows(), post_rows()], ignore_index=True)
    trn = table("tab:sup_training_changes",
                "Training changes tested against the selected configuration of \\diana{} on the dev folds, one at a time, same settings, "
                "seeds and planted mislabels. Each cell gives the paired difference, variant minus \\diana{}, in the share of planted "
                "mislabels caught (mean over the 5, 10 and 20\\,\\% false-flag budgets both models can operate at) and in macro-F1. " + RULE,
                [(None, rows(tc, CHANGES, "DIANA"))])
    screen_grid(tc, [("DIANA", CHANGES, "DIANA")], {"DIANA": "DIANA as selected better", "variant": "change better", "tie": "no difference"},
                FIG / "sup_screens_training_changes.png")
    screen_grid(v, [("DIANA", INPUTS, "DIANA")], {"DIANA": "unitig fraction better", "variant": "alternative input better", "tie": "no difference"},
                FIG / "sup_screens_representations.png")   # DIANA only (Camila 2026-10-23: the baselines are not the object)
    (TABLES / "sup_table_representations.tex").write_text(rep)
    (TABLES / "sup_table_training_changes.tex").write_text(trn)
    n = tc[tc.b == "DIANA"].rule.value_counts().to_dict()
    print("wrote both tables; DIANA cells (inputs + training changes):", n,
          "| logistic regression cells:", v[v.b == "LogReg"].rule.value_counts().to_dict())


if __name__ == "__main__":
    main()
