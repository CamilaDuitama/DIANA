"""Supplementary table: the input-representation screen against the raw fraction.

Source: results/representation_pooled/paired_vs_fraction.tsv (pooled dev-fold OOF,
paired bootstrap over BioProjects). Justifies "no further normalisation or
standardisation is applied" in Methods: 18 arms screened, 60/72 cells tie,
10 worse, 2 better, and the single replicable-looking win (presence/absence on
feature) failed its pre-registered held-out confirmation (G9), so the raw
fraction stayed.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/paper_v2/sup_table_representations.tex"
MANUSCRIPT = Path("/pasteur/helix/scratch/cduitama/EDID/Spinger-Nature-DIANA")

PRETTY_ARM = {
    "abundance": "mean k-mer abundance",
    "presence_absence": "presence/absence",
    "frac_pca192": "fraction, PCA-192",
    "pa_pca192": "presence/absence, PCA-192",
    "frac_pca191nd": "fraction, PCA-191 (depth axis removed)",
    "pa_pca191nd": "presence/absence, PCA-191 (depth axis removed)",
    "frac_plus_kmer4": "fraction + 4-mer composition",
    "frac_plus_clusters": "fraction + unitig clusters",
    "seq_encoder": "sequence encoder input layer",
    "seq_encoder_shuffled": "sequence encoder, shuffled control",
    "attn_pool": "attention pooling input layer",
    "attn_pool_shuffled": "attention pooling, shuffled control",
    "dna_max": "DNA language-model embedding (max)",
    "dna_mean": "DNA language-model embedding (mean)",
    "dna_max_shuffled": "DNA embedding (max), shuffled control",
    "dna_mean_shuffled": "DNA embedding (mean), shuffled control",
    "offlist": "appended off-vocabulary block (standardised)",
    "offlist_shuffled": "off-vocabulary block, shuffled control",
}
TASKS = ["community_type", "feature", "sample_host", "material"]
PRETTY_TASK = {"community_type": "Community type", "feature": "Feature",
               "sample_host": "Sample host", "material": "Material"}


def cell(r) -> str:
    s = f"{r.delta:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
    if r.verdict == "BETTER":
        s += "$^{+}$"
    elif r.verdict == "WORSE":
        s += "$^{-}$"
    return s


def main() -> None:
    d = pd.read_csv(ROOT / "results/representation_pooled/paired_vs_fraction.tsv", sep="\t")
    rows = []
    for arm in PRETTY_ARM:
        sub = d[d.arm == arm].set_index("task")
        cells = [cell(sub.loc[t]) if t in sub.index else "--" for t in TASKS]
        rows.append(f"{PRETTY_ARM[arm]} & " + " & ".join(cells) + " \\\\")
    tex = """\\centering
\\caption{Input representations and transforms screened against the raw unitig fraction (dev folds; pooled out-of-fold macro-F1 over eligible classes; paired difference to the fraction, bootstrapped over BioProjects, 95\\,\\% intervals). $^{+}$/$^{-}$ mark intervals excluding zero in favour of / against the alternative; all other cells are ties. Of 72 cells, 60 tie, 10 are worse and 2 better; the only win considered established (presence/absence on Feature) failed its pre-registered held-out confirmation ($-0.10$ [$-0.15$, $+0.04$]), so the raw fraction remained \\diana's input with no further normalisation.}
\\label{tab:sup_representations}
\\small
\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l""" + "c" * len(TASKS) + """@{}}
\\toprule
Input representation & """ + " & ".join(PRETTY_TASK[t] for t in TASKS) + """ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular*}
"""
    OUT.write_text(tex)
    (MANUSCRIPT / "tables/sup_table_representations.tex").write_text(tex)
    print("wrote", OUT, "and copied to the manuscript")


if __name__ == "__main__":
    main()
