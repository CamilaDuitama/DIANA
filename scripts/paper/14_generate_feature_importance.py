#!/usr/bin/env python3
"""
Generate Feature Importance Figure (Main Figure 3)

PURPOSE:
    Create bar charts showing the top 10 most important taxonomic genera for each
    classification task. Shows which microbial taxa are most discriminative.

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

TOP_N_GENERA = 10
EXCLUDED_TAXONOMY = ['No BLAST hit', 'Unknown taxonomy']


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def generate_feature_importance_figure(output_dir: Path) -> None:
    """Generate feature importance figures showing taxonomic composition by task."""
    # Look for feature importance data
    feature_data_path = Path(PATHS.get('feature_importance_dir', 'results/feature_analysis')) / "feature_importance_by_genus.tsv"
    
    if not feature_data_path.exists():
        print(f"  ⚠ Feature importance data not found at {feature_data_path}")
        print(f"  ⚠ Skipping main_03_feature_importance_*.png")
        print(f"  ℹ Run feature analysis pipeline first to generate this data")
        return
    
    # Load actual feature data
    df = pd.read_csv(feature_data_path, sep='\t')
    
    print(f"✓ Loaded feature importance data: {len(df)} genus-task combinations")
    
    # Generate one figure per task
    for idx, task in enumerate(TASKS):
        # Filter out uninformative taxonomy and get top N
        task_data = df[
            (df['task'] == task) & 
            (~df['genus'].isin(EXCLUDED_TAXONOMY))
        ].nlargest(TOP_N_GENERA, 'n_features')
        
        if len(task_data) == 0:
            print(f"  ⚠ No valid taxonomy data for {task}")
            continue
        
        # Use distinct task-specific color from Vivid palette
        task_color = PLOT_CONFIG['colors']['task_colors'].get(
            task, 
            PLOT_CONFIG['colors']['palette'][idx]
        )
        
        # Create individual figure
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=task_data['n_features'],
                y=task_data['genus'],
                orientation='h',
                marker=dict(
                    color=task_color,
                    opacity=PLOT_CONFIG['fill_opacity'],
                    line=dict(
                        color=PLOT_CONFIG['border_color'], 
                        width=PLOT_CONFIG['line_width']
                    )
                ),
                text=task_data['n_features'],
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title=f"{task.replace('_', ' ').title()} - Taxonomic Composition of Important Features",
            xaxis_title="Number of Important Features",
            yaxis_title="Taxonomy (Genus)",
            template=PLOT_CONFIG['template'],
            font=dict(size=PLOT_CONFIG['font_size']),
            height=600,
            width=800
        )
        
        # Save individual figure
        output_file = output_dir / f"main_03_feature_importance_{task}.png"
        fig.write_html(str(output_file.with_suffix('.html')))
        fig.write_image(str(output_file), width=800, height=600, scale=2)
        
        print(f"  ✓ {task}: Top genus = {task_data.iloc[0]['genus']} ({task_data.iloc[0]['n_features']} features) → {output_file.name}")


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
