#!/usr/bin/env python3
"""
Analyze and visualize unitigs used as input features for the multi-task classifier.

This script:
1. Extracts unitig sequences from unitigs.fa
2. Computes seqkit-like statistics (length distribution, GC content, etc.)
3. Analyzes sparsity/distribution across samples using unitigs.frac.mat
4. Generates publication-ready figures for paper/figures/ and tables for paper/tables/

Usage:
    python scripts/data_prep/06_analyze_unitigs.py \\
        --matrix-dir data/matrices/large_matrix_3070_with_frac
"""

import sys
import argparse
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import logging

# Add project root to path to import diana modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diana.data.loader import MatrixLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_unitig_sequences(fasta_path):
    """Load unitig sequences from FASTA file."""
    logger.info(f"Loading unitig sequences from {fasta_path}...")
    sequences = []
    ids = []
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)
    
    logger.info(f"Loaded {len(sequences)} unitig sequences")
    return ids, sequences


def compute_sequence_stats(sequences):
    """Compute sequence statistics for unitigs."""
    logger.info("Computing sequence statistics...")
    
    stats = {
        'length': [len(seq) for seq in sequences],
        'gc_content': [gc_fraction(seq) * 100 for seq in sequences],
        'n_content': [seq.upper().count('N') / len(seq) * 100 for seq in sequences]
    }
    
    df = pd.DataFrame(stats)
    
    # Summary statistics
    summary = df.describe()
    logger.info(f"\nSequence Statistics:\n{summary}")
    
    return df, summary


def load_fraction_matrix(mat_path):
    """Load fraction matrix using diana.data.loader.MatrixLoader."""
    logger.info(f"Loading fraction matrix from {mat_path}...")
    
    loader = MatrixLoader(mat_path)
    features, sample_ids, _ = loader.load()
    
    # MatrixLoader now auto-transposes: returns (samples x features)
    logger.info(f"Matrix shape: {features.shape} (samples x unitigs)")
    logger.info(f"Sparsity: {100 * (features == 0).sum() / features.size:.2f}%")
    return features, sample_ids


def compute_sparsity_stats(matrix):
    """Compute sparsity and distribution statistics for unitigs across samples."""
    logger.info("Computing sparsity statistics...")
    
    # Matrix is (samples x unitigs), so axis=0 is samples, axis=1 is unitigs
    # Per-unitig statistics (across all samples)
    unitig_stats = {
        'n_samples_present': (matrix > 0).sum(axis=0),  # Sum over samples for each unitig
        'mean_value': matrix.mean(axis=0),
        'max_value': matrix.max(axis=0),
        'std_value': matrix.std(axis=0)
    }
    
    # Per-sample statistics (across all unitigs)
    sample_stats = {
        'n_unitigs_present': (matrix > 0).sum(axis=1),  # Sum over unitigs for each sample
        'mean_value': matrix.mean(axis=1),
    }
    
    # Return as Polars DataFrames for better performance
    return pl.DataFrame(unitig_stats), pl.DataFrame(sample_stats)


def plot_length_distribution(seq_stats_df, output_path):
    """Plot unitig length distribution with plotly."""
    logger.info("Plotting length distribution...")
    
    # Convert to polars for faster operations
    df = pl.DataFrame(seq_stats_df)
    median_length = df['length'].median()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Unitig Length Distribution', 'Unitig Length Box Plot'),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df['length'], nbinsx=50, name='Length', marker_color='steelblue'),
        row=1, col=1
    )
    
    # Add median line
    fig.add_vline(x=median_length, line_dash="dash", line_color="red",
                  annotation_text=f'Median: {median_length:.0f} bp',
                  annotation_position="top right", row=1, col=1)
    
    # Box plot
    fig.add_trace(
        go.Box(y=df['length'], name='Unitigs', marker_color='steelblue'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Unitig Length (bp)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Unitig Length (bp)", row=1, col=2)
    
    fig.update_layout(height=400, width=1200, showlegend=False, template='plotly_white')
    
    # Save interactive HTML and static PNG
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(str(output_path))
    logger.info(f"Saved length distribution plot to {output_path}")


def plot_gc_content(seq_stats_df, output_path):
    """Plot GC content distribution with plotly."""
    logger.info("Plotting GC content distribution...")
    
    df = pl.DataFrame(seq_stats_df)
    median_gc = df['gc_content'].median()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['gc_content'],
        nbinsx=50,
        name='GC Content',
        marker_color='green'
    ))
    
    # Add median line
    fig.add_vline(x=median_gc, line_dash="dash", line_color="red",
                  annotation_text=f'Median: {median_gc:.1f}%',
                  annotation_position="top right")
    
    fig.update_layout(
        title='Unitig GC Content Distribution',
        xaxis_title='GC Content (%)',
        yaxis_title='Frequency',
        height=500,
        width=800,
        template='plotly_white'
    )
    
    # Save interactive HTML and static PNG
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(str(output_path))
    logger.info(f"Saved GC content plot to {output_path}")


def plot_sparsity_distribution(unitig_stats_df, sample_stats_df, output_path):
    """Plot sparsity distribution across samples with plotly."""
    logger.info("Plotting sparsity distribution...")
    
    unitig_df = pl.DataFrame(unitig_stats_df)
    sample_df = pl.DataFrame(sample_stats_df)
    
    # Create 2x2 subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Unitig Presence Across Samples',
            'Unitig Presence Per Sample',
            'Mean Unitig Fraction Across Samples',
            'Maximum Unitig Fraction Across Samples'
        )
    )
    
    # 1. Presence across samples (per unitig)
    median_samples = unitig_df['n_samples_present'].median()
    fig.add_trace(
        go.Histogram(x=unitig_df['n_samples_present'], nbinsx=50, marker_color='steelblue'),
        row=1, col=1
    )
    fig.add_vline(x=median_samples, line_dash="dash", line_color="red",
                  annotation_text=f'Median: {median_samples:.0f}',
                  row=1, col=1)
    
    # 2. Presence across unitigs (per sample)
    median_unitigs = sample_df['n_unitigs_present'].median()
    fig.add_trace(
        go.Histogram(x=sample_df['n_unitigs_present'], nbinsx=50, marker_color='orange'),
        row=1, col=2
    )
    fig.add_vline(x=median_unitigs, line_dash="dash", line_color="red",
                  annotation_text=f'Median: {median_unitigs:.0f}',
                  row=1, col=2)
    
    # 3. Mean fraction values per unitig
    fig.add_trace(
        go.Histogram(x=unitig_df['mean_value'], nbinsx=50, marker_color='purple'),
        row=2, col=1
    )
    
    # 4. Max fraction values per unitig
    fig.add_trace(
        go.Histogram(x=unitig_df['max_value'], nbinsx=50, marker_color='teal'),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Number of Samples Present", row=1, col=1)
    fig.update_yaxes(title_text="Number of Unitigs", row=1, col=1)
    fig.update_xaxes(title_text="Number of Unitigs Present", row=1, col=2)
    fig.update_yaxes(title_text="Number of Samples", row=1, col=2)
    fig.update_xaxes(title_text="Mean Fraction Value", row=2, col=1)
    fig.update_yaxes(title_text="Number of Unitigs", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Maximum Fraction Value", row=2, col=2)
    fig.update_yaxes(title_text="Number of Unitigs", type="log", row=2, col=2)
    
    fig.update_layout(height=1000, width=1200, showlegend=False, template='plotly_white')
    
    # Save interactive HTML and static PNG
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(str(output_path))
    logger.info(f"Saved sparsity distribution plot to {output_path}")


def plot_combined_length_sparsity(seq_stats_df, unitig_stats_df, output_path):
    """Plot relationship between unitig length and sparsity with plotly."""
    logger.info("Plotting length vs sparsity relationship...")
    
    # Merge data
    combined_df = pl.DataFrame({
        'length': seq_stats_df['length'],
        'n_samples': unitig_stats_df['n_samples_present'],
        'mean_frac': unitig_stats_df['mean_value']
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Unitig Length vs Sample Presence', 'Unitig Length vs Mean Fraction')
    )
    
    # Length vs presence
    fig.add_trace(
        go.Scattergl(
            x=combined_df['length'],
            y=combined_df['n_samples'],
            mode='markers',
            marker=dict(size=2, color='steelblue', opacity=0.3),
            name='Unitigs'
        ),
        row=1, col=1
    )
    
    # Length vs mean fraction (for present samples)
    mask = combined_df['mean_frac'] > 0
    filtered_df = combined_df.filter(mask)
    
    fig.add_trace(
        go.Scattergl(
            x=filtered_df['length'],
            y=filtered_df['mean_frac'],
            mode='markers',
            marker=dict(size=2, color='orange', opacity=0.3),
            name='Unitigs with non-zero fraction'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Unitig Length (bp)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Samples Present", row=1, col=1)
    fig.update_xaxes(title_text="Unitig Length (bp)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Fraction Value", type="log", row=1, col=2)
    
    fig.update_layout(height=500, width=1200, showlegend=False, template='plotly_white')
    
    # Save interactive HTML and static PNG
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(str(output_path))
    logger.info(f"Saved length-sparsity relationship plot to {output_path}")


def save_summary_tables(seq_summary, unitig_stats_df, sample_stats_df, output_dir):
    """Save summary statistics tables."""
    logger.info("Saving summary tables...")
    
    # Sequence statistics table
    seq_summary.to_csv(output_dir / "unitig_sequence_stats.csv")
    logger.info(f"Saved sequence statistics to {output_dir / 'unitig_sequence_stats.csv'}")
    
    # Sparsity statistics table
    sparsity_summary = pd.DataFrame({
        'Metric': [
            'Total Unitigs',
            'Total Samples',
            'Median Samples per Unitig',
            'Mean Samples per Unitig',
            'Median Unitigs per Sample',
            'Mean Unitigs per Sample',
            'Overall Sparsity (%)'
        ],
        'Value': [
            len(unitig_stats_df),
            len(sample_stats_df),
            unitig_stats_df['n_samples_present'].median(),
            unitig_stats_df['n_samples_present'].mean(),
            sample_stats_df['n_unitigs_present'].median(),
            sample_stats_df['n_unitigs_present'].mean(),
            100 * (unitig_stats_df['n_samples_present'] == 0).sum() / len(unitig_stats_df)
        ]
    })
    sparsity_summary.to_csv(output_dir / "unitig_sparsity_stats.csv", index=False)
    logger.info(f"Saved sparsity statistics to {output_dir / 'unitig_sparsity_stats.csv'}")
    
    # Top 20 most common unitigs (by sample presence) - using Polars syntax
    top_unitigs = unitig_stats_df.sort('n_samples_present', descending=True).head(20).select([
        'n_samples_present', 'mean_value', 'max_value'
    ])
    top_unitigs.write_csv(output_dir / "top20_common_unitigs.csv")
    logger.info(f"Saved top 20 unitigs to {output_dir / 'top20_common_unitigs.csv'}")


def plot_pca_colored_by_metadata(matrix, sample_ids, metadata_df, output_path):
    """
    Generate PCA plots colored by different metadata categories.
    Creates one separate plot for each category.
    
    Args:
        matrix: Numpy array (samples × features)
        sample_ids: List of sample IDs
        metadata_df: Pandas DataFrame with metadata (must have 'Run_accession' column)
        output_path: Base path for output plots (will create multiple files)
    """
    from sklearn.decomposition import PCA
    
    logger.info("Computing PCA on matrix...")
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(matrix)
    
    # Create DataFrame with PCA coordinates and sample IDs
    pca_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Run_accession': sample_ids
    })
    
    # Merge with metadata
    pca_df = pca_df.merge(metadata_df[['Run_accession', 'sample_type', 'sample_host', 
                                       'community_type', 'material']], 
                         on='Run_accession', how='left')
    
    # Variance explained
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    
    # Create separate plot for each category
    categories = {
        'sample_type': 'Sample Type',
        'sample_host': 'Sample Host',
        'community_type': 'Community Type',
        'material': 'Material'
    }
    
    output_dir = output_path.parent
    base_name = output_path.stem  # e.g., "unitig_pca_by_metadata"
    
    for col_name, display_name in categories.items():
        # Create individual plot
        fig = go.Figure()
        
        # Get unique values for this category
        unique_values = pca_df[col_name].dropna().unique()
        
        # Plot each group
        for value in sorted(unique_values):
            subset = pca_df[pca_df[col_name] == value]
            
            fig.add_trace(
                go.Scattergl(
                    x=subset['PC1'],
                    y=subset['PC2'],
                    mode='markers',
                    name=str(value),
                    marker=dict(size=6, opacity=0.7),
                    showlegend=True
                )
            )
        
        fig.update_xaxes(title_text=f"PC1 ({var_pc1:.1f}%)")
        fig.update_yaxes(title_text=f"PC2 ({var_pc2:.1f}%)")
        
        fig.update_layout(
            title_text=f"PCA: Unitig Matrix Colored by {display_name}",
            height=600,
            width=800,
            showlegend=True,
            legend=dict(
                title=display_name,
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Save with category-specific name
        output_file = output_dir / f"{base_name.replace('_by_metadata', '')}_{col_name}.png"
        fig.write_html(str(output_file).replace('.png', '.html'))
        fig.write_image(output_file, width=800, height=600, scale=2)
        logger.info(f"Saved PCA plot ({display_name}): {output_file}")


def compare_train_test_splits(frac_matrix, sample_ids, train_ids_file, test_ids_file, 
                              seq_stats_df, unitig_stats_df, output_dir):
    """
    Compare unitig distributions between train and test splits.
    
    Args:
        frac_matrix: Numpy array (samples × features)
        sample_ids: List of sample IDs corresponding to matrix rows
        train_ids_file: Path to train sample IDs
        test_ids_file: Path to test sample IDs
        seq_stats_df: DataFrame with sequence statistics
        unitig_stats_df: DataFrame with sparsity statistics
        output_dir: Directory for output plots
    """
    # Load split IDs
    with open(train_ids_file) as f:
        train_ids = set(line.strip() for line in f)
    with open(test_ids_file) as f:
        test_ids = set(line.strip() for line in f)
    
    logger.info(f"Loaded {len(train_ids)} train IDs, {len(test_ids)} test IDs")
    
    # Create boolean masks for train/test samples
    train_mask = np.array([sid in train_ids for sid in sample_ids])
    test_mask = np.array([sid in test_ids for sid in sample_ids])
    
    # Split matrix by train/test
    train_matrix = frac_matrix[train_mask]
    test_matrix = frac_matrix[test_mask]
    
    logger.info(f"Train matrix: {train_matrix.shape}, Test matrix: {test_matrix.shape}")
    
    # Compute statistics for each split
    train_unitig_stats, train_sample_stats = compute_sparsity_stats(train_matrix)
    test_unitig_stats, test_sample_stats = compute_sparsity_stats(test_matrix)
    
    # 1. Compare unitigs per sample
    plot_split_unitigs_per_sample(train_sample_stats, test_sample_stats, 
                                   output_dir / "unitig_split_comparison_per_sample.png")
    
    # 2. Compare mean fraction per unitig
    plot_split_unitig_prevalence(train_unitig_stats, test_unitig_stats,
                                 output_dir / "unitig_split_comparison_prevalence.png")
    
    # 3. Compare unitig presence across splits
    plot_split_unitig_presence(train_unitig_stats, test_unitig_stats,
                               output_dir / "unitig_split_comparison_presence.png")
    
    logger.info("Train/test split comparison complete")


def plot_split_unitigs_per_sample(train_stats, test_stats, output_path):
    """Compare distribution of unitigs per sample between train and test."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Unitigs per Sample", "Box Plot Comparison"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )
    
    # Histogram comparison (convert Polars to numpy for plotly)
    fig.add_trace(
        go.Histogram(x=train_stats['n_unitigs_present'].to_numpy(), name='Train', 
                    opacity=0.7, marker_color='#3b82f6', nbinsx=50),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=test_stats['n_unitigs_present'].to_numpy(), name='Test', 
                    opacity=0.7, marker_color='#ef4444', nbinsx=50),
        row=1, col=1
    )
    
    # Box plot comparison
    fig.add_trace(
        go.Box(y=train_stats['n_unitigs_present'].to_numpy(), name='Train', marker_color='#3b82f6'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=test_stats['n_unitigs_present'].to_numpy(), name='Test', marker_color='#ef4444'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Number of Unitigs", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Number of Unitigs", row=1, col=2)
    
    fig.update_layout(
        title_text="Train vs Test: Unitigs per Sample",
        height=400,
        showlegend=True,
        barmode='overlay'
    )
    
    # Save both formats
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(output_path, width=1200, height=400, scale=2)
    logger.info(f"Saved split comparison (per sample): {output_path}")


def plot_split_unitig_prevalence(train_stats, test_stats, output_path):
    """Compare mean fraction values per unitig between train and test."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Add row number as unitig_id for joining
    train_with_id = train_stats.with_row_count(name='unitig_id')
    test_with_id = test_stats.with_row_count(name='unitig_id')
    
    # Merge on unitig_id using Polars join
    merged = train_with_id.join(test_with_id, on='unitig_id', suffix='_test')
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Mean Fraction Correlation", "Prevalence Distribution"),
        specs=[[{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Scatter: train vs test mean fraction
    fig.add_trace(
        go.Scattergl(
            x=merged['mean_value'].to_numpy(),
            y=merged['mean_value_test'].to_numpy(),
            mode='markers',
            marker=dict(size=2, opacity=0.3, color='#8b5cf6'),
            name='Unitigs',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add diagonal line
    max_val = max(merged['mean_value'].max(), merged['mean_value_test'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='y=x',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Histogram: difference distribution
    diff = (merged['mean_value'] - merged['mean_value_test']).to_numpy()
    fig.add_trace(
        go.Histogram(x=diff, nbinsx=50, marker_color='#8b5cf6'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Train Mean Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Test Mean Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Difference (Train - Test)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    # Calculate correlation
    import numpy as np
    corr = np.corrcoef(merged['mean_value'].to_numpy(), merged['mean_value_test'].to_numpy())[0, 1]
    
    fig.update_layout(
        title_text=f"Train vs Test: Unitig Prevalence (Correlation: {corr:.3f})",
        height=400,
        showlegend=True
    )
    
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(output_path, width=1200, height=400, scale=2)
    logger.info(f"Saved split comparison (prevalence): {output_path}")


def plot_split_unitig_presence(train_stats, test_stats, output_path):
    """Compare number of samples where unitig is present between train and test."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Add row number as unitig_id for joining
    train_with_id = train_stats.with_row_count(name='unitig_id')
    test_with_id = test_stats.with_row_count(name='unitig_id')
    
    # Merge on unitig_id using Polars join
    merged = train_with_id.join(test_with_id, on='unitig_id', suffix='_test')
    
    # Normalize by number of samples in each split
    train_n_samples = train_stats['n_samples_present'].max()
    test_n_samples = test_stats['n_samples_present'].max()
    
    # Add percentage columns using Polars expressions
    merged = merged.with_columns([
        (pl.col('n_samples_present') / train_n_samples * 100).alias('train_pct'),
        (pl.col('n_samples_present_test') / test_n_samples * 100).alias('test_pct')
    ])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Presence % Correlation", "Presence Difference"),
        specs=[[{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Scatter: train vs test presence %
    fig.add_trace(
        go.Scattergl(
            x=merged['train_pct'],
            y=merged['test_pct'],
            mode='markers',
            marker=dict(size=2, opacity=0.3, color='#10b981'),
            name='Unitigs',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 100], y=[0, 100],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='y=x'
        ),
        row=1, col=1
    )
    
    # Histogram: difference distribution
    diff = (merged['train_pct'] - merged['test_pct']).to_numpy()
    fig.add_trace(
        go.Histogram(x=diff, nbinsx=50, marker_color='#10b981'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Train Presence (%)", row=1, col=1)
    fig.update_yaxes(title_text="Test Presence (%)", row=1, col=1)
    fig.update_xaxes(title_text="Difference (Train - Test) %", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    # Calculate correlation
    import numpy as np
    corr = np.corrcoef(merged['train_pct'].to_numpy(), merged['test_pct'].to_numpy())[0, 1]
    
    fig.update_layout(
        title_text=f"Train vs Test: Unitig Presence (Correlation: {corr:.3f})",
        height=400,
        showlegend=True
    )
    
    fig.write_html(str(output_path).replace('.png', '.html'))
    fig.write_image(output_path, width=1200, height=400, scale=2)
    logger.info(f"Saved split comparison (presence): {output_path}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Analyze unitigs and generate visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--matrix-dir', type=Path,
                       default=PROJECT_ROOT / "data" / "matrices" / "large_matrix_3070_with_frac",
                       help='Path to directory containing unitig matrices and sequences')
    parser.add_argument('--matrix-type', type=str,
                       choices=['frac', 'abundance'],
                       required=True,
                       help='Matrix type to analyze: frac or abundance')
    parser.add_argument('--metadata', type=Path,
                       default=PROJECT_ROOT / "data" / "metadata" / "DIANA_metadata.tsv",
                       help='Path to metadata TSV file')
    parser.add_argument('--splits-dir', type=Path,
                       default=PROJECT_ROOT / "data" / "splits",
                       help='Path to directory with train/test split IDs (optional)')
    parser.add_argument('--output-figures', type=Path,
                       default=PROJECT_ROOT / "paper" / "figures" / "data_distribution",
                       help='Base output directory for figures')
    parser.add_argument('--output-tables', type=Path,
                       default=PROJECT_ROOT / "paper" / "tables",
                       help='Base output directory for tables')
    args = parser.parse_args()
    
    # Create matrix-type specific output directories
    matrix_type_suffix = f"{args.matrix_type}_mat"
    figures_dir = args.output_figures / matrix_type_suffix
    tables_dir = args.output_tables / matrix_type_suffix
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info(f"UNITIG ANALYSIS PIPELINE - {args.matrix_type.upper()} MATRIX")
    logger.info("=" * 60)
    logger.info(f"Matrix directory: {args.matrix_dir}")
    logger.info(f"Matrix type: {args.matrix_type}")
    logger.info(f"Metadata: {args.metadata}")
    logger.info(f"Splits directory: {args.splits_dir}")
    logger.info(f"Output figures: {figures_dir}")
    logger.info(f"Output tables: {tables_dir}")
    
    # Input files
    unitigs_fa = args.matrix_dir / "unitigs.fa"
    unitigs_mat = args.matrix_dir / f"unitigs.{args.matrix_type}.mat"
    
    # Load metadata
    if not args.metadata.exists():
        logger.error(f"Metadata file not found: {args.metadata}")
        return
    metadata_df = pd.read_csv(args.metadata, sep='\t')
    
    if not unitigs_fa.exists():
        logger.error(f"Unitigs FASTA not found: {unitigs_fa}")
        return
    if not unitigs_mat.exists():
        logger.error(f"Matrix not found: {unitigs_mat}")
        return
    
    # 1. Load unitig sequences
    unitig_ids, sequences = load_unitig_sequences(unitigs_fa)
    
    # 2. Compute sequence statistics
    seq_stats_df, seq_summary = compute_sequence_stats(sequences)
    
    # 3. Load matrix
    matrix, sample_ids = load_fraction_matrix(unitigs_mat)
    
    # 4. Compute sparsity statistics (full dataset)
    unitig_stats_df, sample_stats_df = compute_sparsity_stats(matrix)
    
    # 5. Generate full dataset plots
    logger.info("Generating full dataset visualizations...")
    plot_length_distribution(seq_stats_df, figures_dir / "unitig_length_distribution.png")
    plot_gc_content(seq_stats_df, figures_dir / "unitig_gc_content.png")
    plot_sparsity_distribution(unitig_stats_df, sample_stats_df, 
                               figures_dir / "unitig_sparsity_distribution.png")
    plot_combined_length_sparsity(seq_stats_df, unitig_stats_df, 
                                   figures_dir / "unitig_length_vs_sparsity.png")
    
    # 6. PCA dimensionality reduction (colored by metadata)
    logger.info("Generating PCA plots...")
    plot_pca_colored_by_metadata(matrix, sample_ids, metadata_df, 
                                 figures_dir / "unitig_pca_by_metadata.png")
    
    # 7. Save summary tables
    save_summary_tables(seq_summary, unitig_stats_df, sample_stats_df, tables_dir)
    
    # 8. Train/test split comparison (if splits available)
    if args.splits_dir and args.splits_dir.exists():
        train_ids_file = args.splits_dir / "train_ids.txt"
        test_ids_file = args.splits_dir / "test_ids.txt"
        
        if train_ids_file.exists() and test_ids_file.exists():
            logger.info("=" * 60)
            logger.info("TRAIN/TEST SPLIT COMPARISON")
            logger.info("=" * 60)
            compare_train_test_splits(matrix, sample_ids, train_ids_file, test_ids_file, 
                                     seq_stats_df, unitig_stats_df, figures_dir)
        else:
            logger.warning(f"Split files not found in {args.splits_dir}, skipping split comparison")
    
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info(f"Tables saved to: {tables_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
