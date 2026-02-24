#!/usr/bin/env python3
"""
Generate Semantic Generalization Sankey Diagram

PURPOSE:
    Visualize how the model generalizes from unseen labels to seen training labels.
    Shows taxonomic hierarchy (subspecies→genus) and semantic similarity.

OUTPUTS:
    - paper/figures/final/main_05_semantic_generalization.png

USAGE:
    python scripts/paper/19_generate_semantic_generalization_sankey.py
    
AUTHOR: Paper generation pipeline
"""

import sys
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

# Add validation scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'validation'))
from load_validation_data import load_validation_predictions

# Add paper config
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS

# ============================================================================
# MANUAL CATEGORIZATION OF GENERALIZATIONS
# ============================================================================

# Define which mappings make sense (correct generalization)
CORRECT_GENERALIZATIONS = {
    'sample_host': {
        # Subspecies → Genus/Species (taxonomic hierarchy)
        ('Gorilla beringei beringei', 'Gorilla sp.'),
        ('Gorilla beringei graueri', 'Gorilla sp.'),
        ('Gorilla gorilla gorilla', 'Gorilla sp.'),
        ('Gorilla beringei', 'Gorilla sp.'),
        ('Pan troglodytes schweinfurthii', 'Pan troglodytes'),
        ('Pan troglodytes ellioti', 'Pan troglodytes'),
    },
    'material': {
        # Specific → General (semantic similarity)
        ('lake sediment', 'sediment'),
        ('marine sediment', 'sediment'),
        ('shallow marine sediment', 'sediment'),
        ('palaeofaeces', 'digestive tract contents'),
        ('brain', 'tissue'),
        ('dentine', 'bone'),
        ('dentine', 'dental calculus'),
    }
}

def load_unseen_mappings():
    """Load validation data and extract unseen label mappings."""
    print("[1/4] Loading validation predictions...")
    df = load_validation_predictions(quiet=True)
    
    # Filter for unseen labels only
    unseen_df = df[~df['is_seen']].copy()
    
    print(f"  ✓ {len(unseen_df)} unseen predictions loaded")
    
    return unseen_df


def create_sankey_diagram(task_df, task_name, correct_mappings, output_file):
    """Create Sankey diagram for a specific task."""
    
    # Get mappings with counts
    mappings = task_df.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    
    # Filter to top N for clarity (or use threshold)
    TOP_N = 15  # Show top 15 mappings
    mappings = mappings.nlargest(TOP_N, 'count')
    
    # Create node labels
    unseen_labels = mappings['true_label'].unique().tolist()
    predicted_labels = mappings['pred_label'].unique().tolist()
    
    # Create all nodes (unseen on left, predicted on right)
    all_labels = unseen_labels + predicted_labels
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    # Create links
    sources = []
    targets = []
    values = []
    colors = []
    
    for _, row in mappings.iterrows():
        true_label = row['true_label']
        pred_label = row['pred_label']
        count = row['count']
        
        source_idx = label_to_idx[true_label]
        target_idx = label_to_idx[pred_label]
        
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(count)
        
        # Color based on whether generalization makes sense
        is_correct = (true_label, pred_label) in correct_mappings
        if is_correct:
            colors.append('rgba(46, 184, 92, 0.6)')  # Green for correct
        else:
            colors.append('rgba(239, 85, 59, 0.4)')  # Red for incorrect
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=all_labels,
            color=['lightblue'] * len(unseen_labels) + ['lightcoral'] * len(predicted_labels),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
        )
    )])
    
    fig.update_layout(
        title=f"{task_name}: Unseen Label Generalizations (Top {TOP_N})",
        font=dict(size=12),
        height=800,
        width=1200,
    )
    
    # Save
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=1200, height=800, scale=2)
    
    print(f"  ✓ {output_file.name}")


def create_combined_sankey(unseen_df, output_dir):
    """Create Sankey diagrams for both tasks."""
    
    print("\n[2/4] Creating Sample Host Sankey...")
    host_df = unseen_df[unseen_df['task'] == 'sample_host']
    if len(host_df) > 0:
        output_file = output_dir / 'main_05_semantic_generalization_host.png'
        create_sankey_diagram(
            host_df,
            'Sample Host',
            CORRECT_GENERALIZATIONS['sample_host'],
            output_file
        )
    
    print("\n[3/4] Creating Material Sankey...")
    mat_df = unseen_df[unseen_df['task'] == 'material']
    if len(mat_df) > 0:
        output_file = output_dir / 'main_05_semantic_generalization_material.png'
        create_sankey_diagram(
            mat_df,
            'Material',
            CORRECT_GENERALIZATIONS['material'],
            output_file
        )


def create_summary_figure(unseen_df, output_dir):
    """Create a summary bar chart showing correct vs incorrect generalizations."""
    
    print("\n[4/4] Creating summary bar chart...")
    
    summary_data = []
    
    for task in ['sample_host', 'material']:
        task_df = unseen_df[unseen_df['task'] == task]
        if len(task_df) == 0:
            continue
        
        # Count correct and incorrect
        correct = 0
        incorrect = 0
        
        for _, row in task_df.iterrows():
            mapping = (row['true_label'], row['pred_label'])
            if mapping in CORRECT_GENERALIZATIONS.get(task, set()):
                correct += 1
            else:
                incorrect += 1
        
        summary_data.append({
            'Task': task.replace('_', ' ').title(),
            'Correct Generalization': correct,
            'Incorrect/Confused': incorrect,
            'Correct %': correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        fig = go.Figure()
        
        # Correct bars
        fig.add_trace(go.Bar(
            name='Correct Generalization',
            x=df_summary['Task'],
            y=df_summary['Correct Generalization'],
            marker_color='rgba(46, 184, 92, 0.8)',
            text=[f"{v} ({p:.1f}%)" for v, p in zip(df_summary['Correct Generalization'], df_summary['Correct %'])],
            textposition='inside',
        ))
        
        # Incorrect bars
        fig.add_trace(go.Bar(
            name='Incorrect/Confused',
            x=df_summary['Task'],
            y=df_summary['Incorrect/Confused'],
            marker_color='rgba(239, 85, 59, 0.8)',
            text=df_summary['Incorrect/Confused'],
            textposition='inside',
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Unseen Label Generalization Performance',
            yaxis_title='Number of Predictions',
            xaxis_title='Task',
            height=500,
            width=800,
            font=dict(size=14),
            showlegend=True,
            legend=dict(x=0.7, y=0.95),
        )
        
        output_file = output_dir / 'main_05_semantic_generalization_summary.png'
        fig.write_html(str(output_file.with_suffix('.html')))
        fig.write_image(str(output_file), width=800, height=500, scale=2)
        
        print(f"  ✓ {output_file.name}")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        for _, row in df_summary.iterrows():
            print(f"{row['Task']:20s}: {row['Correct Generalization']:3.0f} correct ({row['Correct %']:5.1f}%), "
                  f"{row['Incorrect/Confused']:3.0f} confused")


def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING SEMANTIC GENERALIZATION VISUALIZATIONS")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    unseen_df = load_unseen_mappings()
    
    # Create visualizations
    create_combined_sankey(unseen_df, output_dir)
    create_summary_figure(unseen_df, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Semantic generalization figures generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
