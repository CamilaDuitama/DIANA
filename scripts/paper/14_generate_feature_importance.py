#!/usr/bin/env python3
"""
Generate Feature Importance Figure (Main Figure 3)

PURPOSE:
    Create bar charts showing the top 20 most important species for each
    classification task. Shows which microbial taxa are most discriminative.
    "No BLAST hit" and "Other species" are shown in a very light gray.

INPUTS:
    - results/feature_analysis/feature_importance_by_genus.tsv: Feature importance
      grouped by genus taxonomy with columns: task, genus, n_features

OUTPUTS:
    - paper/figures/final/main_03_feature_importance_sample_type.png
    - paper/figures/final/main_03_feature_importance_community_type.png
    - paper/figures/final/main_03_feature_importance_sample_host.png
    - paper/figures/final/main_03_feature_importance_material.png
    - Corresponding .html interactive versions

PROCESS:
    1. Load feature importance data with genus-level taxonomy
    2. For each task:
        a. Filter out uninformative taxonomy ("No BLAST hit", "Unknown taxonomy")
        b. Select top 10 genera by number of important features
        c. Create horizontal bar chart with task-specific color
        d. Save as separate PNG file
    3. Generate 4 individual figures (one per task)

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for inputs/outputs
    - TASKS: List of classification tasks
    - PLOT_CONFIG: Task colors, template, font sizes, borders

HARDCODED VALUES:
    - Top N genera: 10
    - Excluded taxonomy: ["No BLAST hit", "Unknown taxonomy"]
    - Figure size: 800×600 per task

DEPENDENCIES:
    - pandas, plotly
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_feature_importance.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

TOP_N_GENERA = 20
EXCLUDED_TAXONOMY = []  # no exclusions — No BLAST hit shown in light color
LIGHT_COLOR = 'rgba(230, 230, 230, 0.6)'  # near-white for No BLAST hit / Other


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def generate_feature_importance_figure(output_dir: Path) -> None:
    """Generate feature importance figures showing species composition by task."""
    # Use blast_annotations.tsv for species-level data
    blast_path = Path(PATHS.get('feature_importance_dir', 'results/feature_analysis')) / "blast_annotations.tsv"
    feature_data_path = Path(PATHS.get('feature_importance_dir', 'results/feature_analysis')) / "feature_importance_by_genus.tsv"

    if blast_path.exists():
        raw = pd.read_csv(blast_path, sep='\t')
        # Build species column: use best_hit_species, fall back to 'No BLAST hit'
        raw['species'] = raw.apply(
            lambda r: r['best_hit_species']
            if r.get('has_blast_hit', False) and pd.notna(r.get('best_hit_species'))
            else 'No BLAST hit',
            axis=1
        )
        df_source = raw[['task', 'species', 'feature_index']]
        use_species = True
    elif feature_data_path.exists():
        raw = pd.read_csv(feature_data_path, sep='\t')
        raw['species'] = raw['genus']
        df_source = raw.rename(columns={'n_features': 'feature_index'})
        use_species = False
    else:
        print(f"  ⚠ No feature importance data found, skipping")
        return

    print(f"✓ Loaded feature importance data: {len(df_source)} feature-task rows")

    for idx, task in enumerate(TASKS):
        task_data = df_source[df_source['task'] == task].copy()
        if len(task_data) == 0:
            print(f"  ⚠ No data for {task}")
            continue

        # Count features per species
        counts = task_data.groupby('species')['feature_index'].count().reset_index()
        counts.columns = ['species', 'n_features']
        counts = counts.sort_values('n_features', ascending=False)

        # Top 20 named species (excluding No BLAST hit from ranked list)
        named = counts[~counts['species'].isin(['No BLAST hit', 'Unknown taxonomy', 'Uncultured'])]
        top20 = named.head(TOP_N_GENERA).copy()

        # Add No BLAST hit / Unknown as a group at the bottom
        no_hit_n = counts[counts['species'].isin(['No BLAST hit', 'Unknown taxonomy', 'Uncultured'])]['n_features'].sum()
        other_n = named.iloc[TOP_N_GENERA:]['n_features'].sum() if len(named) > TOP_N_GENERA else 0

        rows = []
        if other_n > 0:
            rows.append({'species': f'Other species (>{TOP_N_GENERA})', 'n_features': other_n})
        if no_hit_n > 0:
            rows.append({'species': 'No BLAST hit', 'n_features': no_hit_n})
        extras = pd.DataFrame(rows)

        plot_data = pd.concat([top20, extras], ignore_index=True)
        # Reverse for horizontal bar (bottom = largest)
        plot_data = plot_data.iloc[::-1].reset_index(drop=True)

        task_color = PLOT_CONFIG['colors']['task_colors'].get(
            task, PLOT_CONFIG['colors']['palette'][idx]
        )

        # Assign color per bar
        light_labels = {'No BLAST hit', f'Other species (>{TOP_N_GENERA})'}
        bar_colors = [LIGHT_COLOR if s in light_labels else task_color for s in plot_data['species']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_data['n_features'],
            y=plot_data['species'],
            orientation='h',
            marker=dict(
                color=bar_colors,
                line=dict(color=PLOT_CONFIG['border_color'], width=PLOT_CONFIG['line_width'])
            ),
            text=plot_data['n_features'],
            textposition='outside',
            textfont=dict(size=13)
        ))

        fig.update_layout(
            title=f"{task.replace('_', ' ').title()} — Top {TOP_N_GENERA} Species of Important Features",
            xaxis_title="Number of Important Features",
            yaxis_title="Species",
            template=PLOT_CONFIG['template'],
            font=dict(size=14),
            height=700,
            width=1000,
            margin=dict(l=280)
        )

        output_file = output_dir / f"main_03_feature_importance_{task}.png"
        fig.write_html(str(output_file.with_suffix('.html')))
        fig.write_image(str(output_file), width=1000, height=700, scale=2)
        print(f"  ✓ {task}: Top species = {top20.iloc[-1]['species'] if len(top20) else 'N/A'} → {output_file.name}")


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
