"""
Supplementary Figure 7: BioProject distribution across train, test, and validation splits.

Single-panel figure: horizontal stacked bar chart of the top 30 BioProjects by total
sample count, bars split by Train / Test / Validation.

Also writes paper/tables/final/sup_table_09_bioproject_overlap.tex with the
BioProject overlap statistics across splits.
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import PLOT_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_META = ROOT / "paper/metadata/train_metadata.tsv"
TEST_META  = ROOT / "paper/metadata/test_metadata.tsv"
VAL_META   = ROOT / "paper/metadata/validation_metadata.tsv"
OUT_DIR    = ROOT / "paper/figures/final"
TABLE_DIR  = ROOT / "paper/tables/final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 30
COLORS = {
    "Train":      PLOT_CONFIG["colors"]["train"],
    "Test":       PLOT_CONFIG["colors"]["test"],
    "Validation": PLOT_CONFIG["colors"]["validation"],
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_META, sep="\t", low_memory=False).assign(split="Train")
test  = pd.read_csv(TEST_META,  sep="\t", low_memory=False).assign(split="Test")
val   = pd.read_csv(VAL_META,   sep="\t", low_memory=False).assign(split="Validation")
all_meta = pd.concat([train, test, val], ignore_index=True)

# Keep only rows with a real BioProject code
all_prj = all_meta[all_meta["BioProject"].astype(str).str.startswith("PRJ")].copy()

# ---------------------------------------------------------------------------
# Top-N BioProjects by total count
# ---------------------------------------------------------------------------
total_counts = all_prj.groupby("BioProject").size().sort_values(ascending=False)
top_projects = total_counts.head(TOP_N).index.tolist()
top_projects_display = list(reversed(top_projects))  # largest at top in hbar

split_counts = (
    all_prj[all_prj["BioProject"].isin(top_projects)]
    .groupby(["BioProject", "split"])
    .size()
    .unstack(fill_value=0)
)

# ---------------------------------------------------------------------------
# BioProject overlap statistics (for table)
# ---------------------------------------------------------------------------
bp_splits = all_prj.groupby("BioProject")["split"].apply(set)
n_total    = len(bp_splits)
stats = {
    "Train only":       int((bp_splits == {"Train"}).sum()),
    "Test only":        int((bp_splits == {"Test"}).sum()),
    "Validation only":  int((bp_splits == {"Validation"}).sum()),
    "Train + Test":     int(bp_splits.apply(lambda s: s >= {"Train", "Test"} and "Validation" not in s).sum()),
    "Train + Val":      int(bp_splits.apply(lambda s: s >= {"Train", "Validation"} and "Test" not in s).sum()),
    "Test + Val":       int(bp_splits.apply(lambda s: s >= {"Test", "Validation"} and "Train" not in s).sum()),
    "All three splits": int(bp_splits.apply(lambda s: s == {"Train", "Test", "Validation"}).sum()),
}

# ---------------------------------------------------------------------------
# Figure: single-panel horizontal stacked bar
# ---------------------------------------------------------------------------
fig = go.Figure()

for split in ["Train", "Test", "Validation"]:
    if split not in split_counts.columns:
        continue
    values = [split_counts.loc[bp, split] if bp in split_counts.index else 0
              for bp in top_projects_display]
    fig.add_trace(
        go.Bar(
            name=split,
            y=top_projects_display,
            x=values,
            orientation="h",
            marker=dict(color=COLORS[split], line=dict(color="white", width=0.5)),
        )
    )

fig.update_layout(
    template=PLOT_CONFIG["template"],
    font=dict(size=PLOT_CONFIG["font_size"]),
    height=750,
    width=950,
    barmode="stack",
    legend=dict(title="Split", orientation="v"),
    xaxis_title="Number of samples",
    yaxis_title="BioProject",
    yaxis=dict(tickfont=dict(size=10)),
    margin=dict(l=20, r=20, t=40, b=60),
    title=f"Top {TOP_N} BioProjects by sample count ({n_total} total)",
)

html_out = OUT_DIR / "sup_07_bioproject_distribution.html"
png_out  = OUT_DIR / "sup_07_bioproject_distribution.png"
fig.write_html(str(html_out))
fig.write_image(str(png_out), scale=2)
print(f"✓ {html_out.name}")
print(f"✓ {png_out.name}")

# ---------------------------------------------------------------------------
# LaTeX table: BioProject overlap
# ---------------------------------------------------------------------------
rows = ""
for label, count in stats.items():
    rows += f"    {label} & {count} \\\\\n"

latex = r"""\begin{table}[h]
\centering
\caption{\textbf{BioProject overlap across dataset splits.}
Number of BioProjects exclusive to each split or shared across splits,
out of """ + str(n_total) + r""" distinct BioProjects in total.}
\label{tab:bioproject_overlap}
\begin{tabular}{lc}
\toprule
\textbf{Split membership} & \textbf{BioProjects (n)} \\
\midrule
""" + rows + r"""\bottomrule
\end{tabular}
\end{table}
"""

table_out = TABLE_DIR / "sup_table_09_bioproject_overlap.tex"
table_out.write_text(latex)
print(f"✓ {table_out.name}")
