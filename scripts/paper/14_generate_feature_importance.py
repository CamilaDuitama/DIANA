#!/usr/bin/env python3
"""
Generate Feature Importance Figure (Main Figure 3)

PURPOSE:
    Lollipop plots showing the top 15 most important unitig features per task,
    annotated with their BLAST species hit. Each point is colored by annotation
    category (oral, environmental, host/eukaryote, MAG, uncultured/no annotation).

INPUTS:
    - paper/tables/feature_analysis/top_features_{task}_with_sequences.csv
      (rank, unitig id, importance_score)
    - results/feature_analysis/unitigs_with_blast_hits.tsv
      (unitig_id -> blast_description, has_blast_hit)
    - results/feature_analysis/logan_best_hits_top_features.tsv
      (unitig_id -> sgenome, pident, scientific_name) — fallback for no-BLAST features

OUTPUTS:
    - paper/figures/final/main_03_feature_importance_{task}.png/.html  (×4)

DEPENDENCIES:
    - pandas, plotly, re
    - config.py (same directory)
"""

import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG


# ============================================================================
# PARAMETERS
# ============================================================================

TOP_N = 15  # top features per task

# Annotation categories and their colors
CATEGORY_COLORS = {
    'Oral bacteria':        '#E45756',
    'Environmental bacteria': '#72B7B2',
    'Host / eukaryote':     '#F58518',
    'MAG / metagenome':     '#54A24B',
    'Uncultured':           '#B279A2',
    'No annotation':        '#BBBFC1',
}


# ============================================================================
# HELPERS
# ============================================================================

def clean_label(desc: str) -> str:
    """Return a short, readable species label from a BLAST description."""
    if pd.isna(desc):
        return 'No annotation'
    d = str(desc)
    d = re.sub(r'^MAG:\s*', '', d)
    d = re.sub(r'\s+(isolate|strain|clone|chromosome|genome|partial|complete|16S|small subunit).*$',
               '', d, flags=re.IGNORECASE)
    d = re.sub(r'^PREDICTED:\s*', '', d)
    return d.strip()[:60]


# Keywords used to assign annotation category
_ORAL = {'streptococcus', 'cutibacterium', 'neisseria', 'aggregatibacter',
         'lactococcus', 'arachnia', 'corynebacterium', 'selenomonas',
         'actinomyces', 'schaalia', 'fusobacterium', 'desulfobulbus oralis',
         'rothia', 'veillonella', 'prevotella', 'porphyromonas', 'tannerella',
         'treponema', 'capnocytophaga', 'actinobacillus', 'ottowia',
         'mediterraneibacter', 'acinetobacter'}
_HOST = {'homo sapiens', 'ailuropoda', 'mus musculus', 'bos taurus',
         'sus scrofa', 'canis lupus', 'felis catus', 'sparganium', 'crambe',
         'hippidion', 'clupea', 'equus', 'cervus', 'bison', 'ovis', 'capra',
         'gallus', 'danio', 'drosophila', 'arabidopsis'}
_UNCULTURED = {'uncultured', 'uncultur'}


def annotate_category(desc: str) -> str:
    if pd.isna(desc):
        return 'No annotation'
    d = desc.lower()
    if any(k in d for k in _HOST):
        return 'Host / eukaryote'
    if any(k in d for k in _ORAL):
        return 'Oral bacteria'
    if any(k in d for k in _UNCULTURED):
        return 'Uncultured'
    if 'mag:' in desc.lower() or 'mag ' in desc.lower():
        return 'MAG / metagenome'
    return 'Environmental bacteria'


# ============================================================================
# MAIN PLOTTING FUNCTION
# ============================================================================

def generate_feature_importance_figure(output_dir: Path) -> None:
    """Generate lollipop feature importance figures, one per task."""
    feat_dir = Path('paper/tables/feature_analysis')
    blast_path = Path('results/feature_analysis/unitigs_with_blast_hits.tsv')

    if not blast_path.exists():
        print(f"  ⚠ {blast_path} not found, skipping")
        return
    blast = pd.read_csv(blast_path, sep='\t').set_index('unitig_id')
    print(f"✓ Loaded BLAST annotations: {blast['has_blast_hit'].sum():,} / {len(blast):,} unitigs with hits")

    logan_path = Path('results/feature_analysis/logan_best_hits_top_features.tsv')
    if logan_path.exists():
        logan = pd.read_csv(logan_path, sep='\t').set_index('unitig_id')
        print(f"✓ Loaded Logan fallback: {len(logan):,} unitigs")
    else:
        logan = pd.DataFrame(columns=['sgenome', 'pident', 'qcovHSP', 'scientific_name'])
        print(f"  ⚠ Logan fallback file not found, skipping Logan annotations")

    for idx, task in enumerate(TASKS):
        feat_file = feat_dir / f'top_features_{task}_with_sequences.csv'
        if not feat_file.exists():
            print(f"  ⚠ {feat_file} not found, skipping {task}")
            continue

        df = pd.read_csv(feat_file).head(TOP_N)

        labels, scores, categories = [], [], []
        for _, row in df.iterrows():
            uid = row['id']
            score = row['importance_score']
            if uid in blast.index and blast.loc[uid, 'has_blast_hit']:
                desc = blast.loc[uid, 'blast_description']
                label = clean_label(desc)
                cat = annotate_category(desc)
            elif uid in logan.index:
                sci = str(logan.loc[uid, 'scientific_name'])
                label = f'{sci} †'
                cat = annotate_category(sci.lower())
            else:
                label = 'No annotation'
                cat = 'No annotation'
            labels.append(label)
            scores.append(score)
            categories.append(cat)

        # Reverse so rank 1 is at the top
        labels = labels[::-1]
        scores = scores[::-1]
        categories = categories[::-1]

        task_title = task.replace('_', ' ').title()

        fig = go.Figure()

        # Horizontal lines (stems)
        for i, (s, cat) in enumerate(zip(scores, categories)):
            fig.add_shape(
                type='line',
                x0=0, x1=s, y0=i, y1=i,
                line=dict(color=CATEGORY_COLORS[cat], width=2)
            )

        # Dots grouped by category for legend
        seen_cats = set()
        for cat, color in CATEGORY_COLORS.items():
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            if not cat_indices:
                continue
            fig.add_trace(go.Scatter(
                x=[scores[i] for i in cat_indices],
                y=cat_indices,
                mode='markers',
                marker=dict(size=12, color=color,
                            line=dict(color='white', width=1.5)),
                name=cat,
                showlegend=True,
            ))

        fig.update_layout(
            title=task_title,
            xaxis_title='Importance score',
            yaxis=dict(
                tickvals=list(range(TOP_N)),
                ticktext=labels,
                tickfont=dict(size=16),
                title=None,
            ),
            template=PLOT_CONFIG['template'],
            font=dict(size=18),
            title_font_size=24,
            height=750,
            width=1100,
            margin=dict(l=360, r=20, t=70, b=80),
            legend=dict(
                title='Annotation',
                title_font_size=17,
                font=dict(size=16),
                itemsizing='constant',
                bgcolor='rgba(255,255,255,0.85)',
            ),
            xaxis=dict(
                title_font_size=20,
                tickfont=dict(size=16),
                rangemode='tozero',
            ),
        )

        output_file = output_dir / f'main_03_feature_importance_{task}.png'
        fig.write_html(str(output_file.with_suffix('.html')))
        fig.write_image(str(output_file), width=1100, height=750, scale=2)
        print(f"  ✓ {task} → {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING FEATURE IMPORTANCE FIGURE")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    print("\nGenerating feature importance visualization...")
    generate_feature_importance_figure(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Feature importance figures generated (4 separate plots)")
    print("=" * 80)


if __name__ == '__main__':
    main()
