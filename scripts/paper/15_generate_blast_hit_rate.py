#!/usr/bin/env python3
"""
Generate BLAST Hit Rate Figure (Main Figure 4)

PURPOSE:
    Compare BLAST hit rates between important discriminative features and all features.
    Shows that important features have higher annotation rates, suggesting they represent
    known microbial sequences.

INPUTS:
    - results/feature_analysis/blast_annotations.tsv: BLAST results for important features
      with columns: task, has_blast_hit (or has_hit)
    - results/feature_analysis/all_features_blast/blast_summary.json: Overall hit rate
      for all 107,480 unitigs with key: hit_rate_percent

OUTPUTS:
    - paper/figures/final/main_04_blast_hit_rate.png
    - paper/figures/final/main_04_blast_hit_rate.html

PROCESS:
    1. Load BLAST annotations for important features (top 100 per task)
    2. Calculate hit rate per task for important features
    3. Load overall hit rate for all features from summary JSON
    4. Create grouped bar chart:
        - X-axis: Tasks (sample_type, community_type, sample_host, material)
        - Y-axis: BLAST hit rate (%)
        - Two bars per task: Important features vs All features
    5. Save as high-resolution PNG and interactive HTML

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for inputs/outputs
    - TASKS: List of classification tasks
    - PLOT_CONFIG: Colors (Blue/Orange), template, borders

HARDCODED VALUES:
    - Feature set labels: "Important Features (Top 100 per task)", "All Features (107,480 unitigs)"
    - Y-axis range: 0-105%
    - Task order: sample_type, community_type, sample_host, material

DEPENDENCIES:
    - pandas, plotly, json
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_blast_hit_rate.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, TASKS, PLOT_CONFIG


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

IMPORTANT_FEATURES_LABEL = "Important Features (Top 100 per task)"
ALL_FEATURES_LABEL = "All Features (107,480 unitigs)"
Y_AXIS_RANGE = [0, 105]  # 0-100% with padding


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def generate_blast_hit_rate_figure(output_dir: Path) -> None:
    """Generate BLAST hit rate comparison: ALL features vs most important features."""
    
    # Load important features BLAST results
    blast_important_path = Path(PATHS.get('blast_results', 'results/feature_analysis/blast_annotations.tsv'))
    
    # Load all features BLAST summary
    blast_all_summary_path = Path("results/feature_analysis/all_features_blast/blast_summary.json")
    
    if not blast_important_path.exists():
        print(f"  ⚠ BLAST results not found at {blast_important_path}")
        print(f"  ⚠ Skipping main_04_blast_hit_rate.png")
        print(f"  ℹ Run BLAST annotation pipeline first")
        return
    
    # Calculate hit rate for important features (per task)
    df_important = pd.read_csv(blast_important_path, sep='\t')
    print(f"✓ Loaded BLAST annotations: {len(df_important)} important features")
    
    # Find the hit column (handle different naming)
    hit_col = 'has_blast_hit' if 'has_blast_hit' in df_important.columns else 'has_hit'
    
    # Group by task for important features
    if 'task' in df_important.columns:
        task_hit_rates = df_important.groupby('task')[hit_col].mean() * 100
        print(f"✓ Calculated hit rates per task")
    else:
        task_hit_rates = pd.Series({'Overall': df_important[hit_col].mean() * 100})
        print(f"✓ Calculated overall hit rate (no task column found)")
    
    # Load overall hit rate for ALL features
    all_features_hit_rate = None
    if blast_all_summary_path.exists():
        with open(blast_all_summary_path) as f:
            all_summary = json.load(f)
        all_features_hit_rate = all_summary.get('hit_rate_percent', 0)
        print(f"✓ All features hit rate: {all_features_hit_rate:.2f}%")
    else:
        print(f"  ⚠ All features BLAST summary not found at {blast_all_summary_path}")
        print(f"  ℹ Showing only important features hit rates")
    
    # Create grouped bar chart
    fig = go.Figure()
    
    task_labels = [t.replace('_', ' ').title() for t in TASKS]
    
    # Important features hit rates per task
    important_rates = [task_hit_rates.get(task, 0) for task in TASKS]
    
    fig.add_trace(go.Bar(
        x=task_labels,
        y=important_rates,
        name=IMPORTANT_FEATURES_LABEL,
        marker=dict(
            color=PLOT_CONFIG['colors']['palette'][0],  # Blue
            opacity=PLOT_CONFIG['fill_opacity'],
            line=dict(
                color=PLOT_CONFIG['border_color'], 
                width=PLOT_CONFIG['line_width']
            )
        ),
        text=[f"{rate:.1f}%" for rate in important_rates],
        textposition='outside'
    ))
    
    # All features hit rate (constant across tasks)
    if all_features_hit_rate is not None:
        fig.add_trace(go.Bar(
            x=task_labels,
            y=[all_features_hit_rate] * len(TASKS),
            name=ALL_FEATURES_LABEL,
            marker=dict(
                color=PLOT_CONFIG['colors']['palette'][1],  # Orange
                opacity=PLOT_CONFIG['fill_opacity'],
                line=dict(
                    color=PLOT_CONFIG['border_color'], 
                    width=PLOT_CONFIG['line_width']
                )
            ),
            text=[f"{all_features_hit_rate:.1f}%"] * len(TASKS),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='BLAST Hit Rate: Important Features vs All Features',
        xaxis_title='Task',
        yaxis_title='BLAST Hit Rate (%)',
        template=PLOT_CONFIG['template'],
        height=600,
        width=900,
        font=dict(size=PLOT_CONFIG['font_size']),
        barmode='group',
        yaxis=dict(range=Y_AXIS_RANGE),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save outputs
    output_file = output_dir / "main_04_blast_hit_rate.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=900, height=600, scale=2)
    
    print(f"  ✓ {output_file.name}")
    
    # Print summary statistics
    print("\nHit rate summary:")
    for task, rate in zip(TASKS, important_rates):
        print(f"  {task}: {rate:.1f}% (important features)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING BLAST HIT RATE FIGURE")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    print("\nGenerating BLAST hit rate comparison...")
    generate_blast_hit_rate_figure(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - BLAST hit rate figure generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
