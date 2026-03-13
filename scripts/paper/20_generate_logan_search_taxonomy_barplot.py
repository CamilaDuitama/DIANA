#!/usr/bin/env python3
"""
Generate Logan Search Taxonomy Barplot (Supplementary Figure)

PURPOSE:
    Visualise the taxonomic/biological origin of the SRA runs that are the
    best Logan-database hits for the 24,627 BLAST-unannotated DIANA unitig
    features.  This demonstrates that these "dark" unitigs are genuine
    biological sequences widely found in human-associated microbiomes, and
    that they represent no data leakage from DIANA itself (all source runs
    are external to DIANA).

INPUT ANNOTATION APPROACH:
    The ENA metadata file (`data/metadata/logan_best_hit_sra_metadata.tsv`)
    was fetched from the ENA Portal API
    (https://www.ebi.ac.uk/ena/portal/api/filereport?result=read_run&fields=all)
    one accession at a time for the 886 SRA runs that are the best Logan hit
    (highest bitscore) per unitig and that do not belong to DIANA.

    The column used for taxonomic annotation is **`scientific_name`** — the
    NCBI taxonomic name of the organism/environment associated with each SRA
    run (e.g. "Homo sapiens", "human oral metagenome", "Mycobacterium
    tuberculosis").  The top 10 most frequent scientific names are shown
    individually; all remaining names are collapsed into "Others".

INPUTS:
    - data/metadata/logan_best_hit_sra_metadata_resolved.tsv  (preferred)
    - data/metadata/logan_best_hit_sra_metadata.tsv           (fallback)
        ENA metadata (196 columns) for the best-hit SRA runs, fetched by
        scripts/feature_analysis/06_fetch_logan_hit_sra_metadata.py and
        07_refetch_uninformative_logan_hits.py.
        Key column: scientific_name (NCBI taxa/environment label).

OUTPUTS:
    - paper/figures/final/sup_05_logan_taxonomy_barplot.png  (bar chart)
    - paper/figures/final/sup_05_logan_taxonomy_barplot.html
    - paper/figures/final/sup_05_logan_coverage_piechart.png  (pie chart)
    - paper/figures/final/sup_05_logan_coverage_piechart.html

DEPENDENCIES:
    - pandas, plotly
    - config.py (same directory)

USAGE:
    python scripts/paper/20_generate_logan_search_taxonomy_barplot.py

AUTHOR: Generated for DIANA paper supplementary materials.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

# Use resolved metadata (with fallback for uninformative entries) when available
METADATA_PATH = Path("data/metadata/logan_best_hit_sra_metadata_resolved.tsv")
METADATA_PATH_RAW = Path("data/metadata/logan_best_hit_sra_metadata.tsv")

# Known totals from BLAST and Logan analyses
TOTAL_UNITIGS   = 107_480   # all DIANA unitig features
BLAST_HITS      = 82_853    # unitigs with ≥1 BLAST hit (NCBI nt)
LOGAN_ONLY_HITS = 24_596    # no BLAST hit, but ≥1 Logan hit
NO_HIT          = 31        # no BLAST hit AND no Logan hit

TOP_N        = 20           # Number of most-frequent scientific names to show individually
OTHERS_LABEL = "Others"


# ============================================================================
# PIE CHART
# ============================================================================

def generate_coverage_piechart(output_dir: Path) -> None:
    """Pie chart: BLAKE hits vs Logan-only hits vs unannotated, for all 107,480 unitigs."""

    labels = [
        f"BLAST hit<br>(NCBI nt)",
        f"Logan hit only<br>(SRA index)",
        f"No hit",
    ]
    values = [BLAST_HITS, LOGAN_ONLY_HITS, NO_HIT]
    pcts   = [round(v / TOTAL_UNITIGS * 100, 1) for v in values]
    palette = PLOT_CONFIG["colors"]["palette"]
    colors  = [palette[0], palette[2], palette[1]]

    print("\nCoverage pie chart:")
    for l, v, p in zip(labels, values, pcts):
        print(f"  {v:6,d}  ({p:.1f}%)  {l.replace(chr(10), ' ')}")

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            text=[f"{v:,}<br>({p}%)" for v, p in zip(values, pcts)],
            textinfo="text",
            textposition="outside",
            hovertemplate="%{label}: %{value:,} unitigs (%{percent})<extra></extra>",
            marker=dict(
                colors=colors,
                line=dict(
                    color=PLOT_CONFIG["border_color"],
                    width=PLOT_CONFIG["line_width"],
                ),
            ),
            opacity=PLOT_CONFIG["fill_opacity"],
            sort=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Sequence coverage of all {TOTAL_UNITIGS:,} DIANA unitig features",
            x=0.5,
            font_size=15,
        ),
        template=PLOT_CONFIG["template"],
        font=dict(size=PLOT_CONFIG["font_size"]),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(t=100, b=80, l=40, r=40),
        width=550,
        height=480,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "sup_05_logan_coverage_piechart.html"
    png_path  = output_dir / "sup_05_logan_coverage_piechart.png"

    fig.write_html(str(html_path))
    print(f"Saved HTML → {html_path}")
    try:
        fig.write_image(str(png_path), scale=3)
        print(f"Saved PNG  → {png_path}")
    except Exception as e:
        print(f"[WARN] Could not save PNG: {e}")


# ============================================================================
# MAIN
# ============================================================================

def generate_logan_taxonomy_barplot(output_dir: Path) -> None:
    """Load ENA metadata, show top-N scientific names + Others as a barplot."""

    # ── load ──────────────────────────────────────────────────────────────
    path = METADATA_PATH if METADATA_PATH.exists() else METADATA_PATH_RAW
    if not path.exists():
        raise FileNotFoundError(
            f"ENA metadata not found at {METADATA_PATH} or {METADATA_PATH_RAW}.\n"
            "Run scripts/feature_analysis/06_fetch_logan_hit_sra_metadata.py first."
        )
    print(f"Using metadata: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Normalise missing scientific names to a placeholder
    df["scientific_name"] = df["scientific_name"].fillna("").str.strip()
    df.loc[df["scientific_name"] == "", "scientific_name"] = "Unknown"
    total = len(df)
    print(f"Loaded {total} runs")

    # ── top-N + Others ────────────────────────────────────────────────────
    counts = df["scientific_name"].value_counts()
    top = counts.iloc[:TOP_N]
    others_count = counts.iloc[TOP_N:].sum()

    labels = list(top.index) + [OTHERS_LABEL]
    values = list(top.values) + [others_count]
    pcts   = [round(v / total * 100, 1) for v in values]

    print(f"\nTop {TOP_N} scientific names + Others:")
    for label, cnt, pct in zip(labels, values, pcts):
        print(f"  {cnt:4d}  ({pct:.1f}%)  {label}")

    # ── figure ────────────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=PLOT_CONFIG["colors"]["palette"][0],
                opacity=PLOT_CONFIG["fill_opacity"],
                line=dict(
                    color=PLOT_CONFIG["border_color"],
                    width=PLOT_CONFIG["line_width"],
                ),
            ),
            text=[f"{v}<br>({p}%)" for v, p in zip(values, pcts)],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig.update_layout(
        title=dict(
            text="Taxonomic origin of Logan search hits",
            x=0.5,
            font_size=15,
        ),
        xaxis=dict(
            title="Organism / environment (ENA)",
            tickangle=-35,
            tickfont_size=11,
        ),
        yaxis=dict(
            title="Number of SRA runs",
            range=[0, max(values) * 1.22],
        ),
        template=PLOT_CONFIG["template"],
        font=dict(size=PLOT_CONFIG["font_size"]),
        bargap=0.3,
        margin=dict(t=120, b=160, l=70, r=40),
        width=1000,
        height=580,
        showlegend=False,
    )

    # ── save ──────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "sup_05_logan_taxonomy_barplot.html"
    png_path  = output_dir / "sup_05_logan_taxonomy_barplot.png"

    fig.write_html(str(html_path))
    print(f"\nSaved HTML → {html_path}")

    try:
        fig.write_image(str(png_path), scale=3)
        print(f"Saved PNG  → {png_path}")
    except Exception as e:
        print(f"[WARN] Could not save PNG (kaleido not installed?): {e}")
        print("       HTML figure saved successfully.")


if __name__ == "__main__":
    output_dir = Path(PATHS.get("figures_dir", "paper/figures/final"))
    generate_coverage_piechart(output_dir)
    generate_logan_taxonomy_barplot(output_dir)
