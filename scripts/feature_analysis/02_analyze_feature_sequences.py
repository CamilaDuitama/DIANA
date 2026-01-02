#!/usr/bin/env python3
"""
Analyze Feature Sequences - Map Important Features to Unitig Sequences
=======================================================================

Takes the top important features and analyzes their unitig sequences:
- GC content distribution
- Sequence length distribution  
- Complexity analysis
- Comparison across tasks

OUTPUT:
-------
- Sequence statistics for top features
- Interactive plots (HTML + PNG)
- Summary tables with sequences

USAGE:
------
python scripts/feature_analysis/02_analyze_feature_sequences.py \\
    --config configs/feature_analysis.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import logging

from diana.data.unitig_analyzer import UnitigAnalyzer
from diana.data.loader import MatrixLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_feature_importance_tables(tables_dir: Path, method: str = 'weight_based') -> pl.DataFrame:
    """Load feature importance tables."""
    table_path = tables_dir / f'top_100_features_{method}.csv'
    
    if not table_path.exists():
        raise FileNotFoundError(f"Feature importance table not found: {table_path}")
    
    df = pl.read_csv(table_path)
    logger.info(f"Loaded {len(df)} feature importance records from {table_path}")
    return df


def load_blast_annotations(tables_dir: Path) -> pl.DataFrame:
    """Load BLAST annotations from annotated feature tables."""
    all_annotations = []
    
    # Load all task-specific annotated tables
    for annotated_file in sorted(tables_dir.glob('top_features_*_annotated.csv')):
        try:
            df = pl.read_csv(annotated_file)
            if 'best_hit_species' in df.columns:
                # Extract task name from filename
                task = annotated_file.stem.replace('top_features_', '').replace('_annotated', '')
                df = df.with_columns(pl.lit(task).alias('task'))
                all_annotations.append(df)
        except Exception as e:
            logger.warning(f"Could not load {annotated_file}: {e}")
    
    if all_annotations:
        combined = pl.concat(all_annotations)
        logger.info(f"Loaded BLAST annotations for {len(combined)} features")
        return combined
    else:
        logger.warning("No BLAST annotations found")
        return pl.DataFrame()


def compute_fraction_statistics(X_frac: np.ndarray, feature_indices: List[int]) -> pl.DataFrame:
    """
    Compute fraction distribution statistics for features.
    
    For each feature, compute:
    - Prevalence: % of samples where feature is present (>0)
    - Mean fraction (all samples)
    - Mean fraction (samples where present)
    - Median fraction (samples where present)
    - Min/Max fraction (samples where present)
    """
    logger.info(f"Computing fraction statistics for {len(feature_indices)} features...")
    
    stats = []
    for feat_idx in feature_indices:
        feat_vals = X_frac[:, feat_idx]
        present_mask = feat_vals > 0
        n_present = present_mask.sum()
        prevalence = 100 * n_present / len(feat_vals)
        
        mean_all = feat_vals.mean()
        
        if n_present > 0:
            present_vals = feat_vals[present_mask]
            mean_present = present_vals.mean()
            median_present = np.median(present_vals)
            min_present = present_vals.min()
            max_present = present_vals.max()
        else:
            mean_present = 0.0
            median_present = 0.0
            min_present = 0.0
            max_present = 0.0
        
        stats.append({
            'feature_index': feat_idx,
            'prevalence_pct': prevalence,
            'n_samples_present': n_present,
            'mean_frac_all': mean_all,
            'mean_frac_present': mean_present,
            'median_frac_present': median_present,
            'min_frac_present': min_present,
            'max_frac_present': max_present
        })
    
    return pl.DataFrame(stats)


def create_sequence_analysis_plots(
    df_merged: pl.DataFrame,
    task_names: List[str],
    output_dir: Path,
    top_k: int = 50
):
    """Create sequence analysis visualizations."""
    logger.info("Creating sequence analysis plots...")
    
    # 1. GC content distribution for top features per task
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[task.replace('_', ' ').title() for task in task_names],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    for idx, task in enumerate(task_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        task_data = df_merged.filter(
            (pl.col('task') == task) & (pl.col('rank') <= top_k)
        ).sort('rank')
        # Build customdata for hover: species, genus, phylum
        customdata = []
        for row_ in task_data.iter_rows(named=True):
            customdata.append([
                row_.get('best_hit_species', ''),
                row_.get('genus', ''),
                row_.get('phylum', ''),
                row_.get('prevalence_pct', None)
            ])
        fig.add_trace(
            go.Scatter(
                x=task_data['rank'].to_list(),
                y=task_data['gc_content'].to_list(),
                mode='markers',
                marker=dict(
                    size=8,
                    color=task_data['importance_score'].to_list(),
                    colorscale='Viridis',
                    showscale=(idx == 0),
                    colorbar=dict(title='Importance', x=1.15) if idx == 0 else None,
                    line=dict(width=0.5, color='white')
                ),
                text=[f"F{i}" for i in task_data['feature_index'].to_list()],
                customdata=customdata,
                hovertemplate='<b>Feature %{text}</b><br>' +
                             'Rank: %{x}<br>' +
                             'GC: %{y:.1f}%<br>' +
                             'Species: %{customdata[0]}<br>' +
                             'Genus: %{customdata[1]}<br>' +
                             'Phylum: %{customdata[2]}<br>' +
                             'Prevalence: %{customdata[3]:.1f}%<br>' +
                             '<extra></extra>',
                name=task,
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add median line
        median_gc = task_data['gc_content'].median()
        fig.add_hline(
            y=median_gc,
            line_dash="dash",
            line_color="red",
            opacity=0.5,
            row=row, col=col
        )
        
        fig.update_xaxes(title_text='Feature Rank', row=row, col=col)
        fig.update_yaxes(title_text='GC Content (%)', row=row, col=col)
    
    fig.update_layout(
        title_text='GC Content of Top Features by Task',
        height=800,
        width=1200,
        template='plotly_white'
    )
    
    html_path = output_dir / 'sequence_gc_content_by_task.html'
    png_path = output_dir / 'sequence_gc_content_by_task.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=1200, height=800, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")
    
    # 2. Sequence complexity vs importance
    plot_data = df_merged.filter(pl.col('rank') <= top_k).to_pandas()
    # Build custom hover text
    hover_text = []
    for _, row_ in plot_data.iterrows():
        parts = [
            f"Feature: {row_['feature_index']}",
            f"Rank: {row_['rank']}",
            f"Length: {row_['length']} bp",
            f"GC: {row_['gc_content']:.1f}%",
            f"Complexity: {row_['complexity']:.3f}",
            f"Importance: {row_['importance_score']:.4f}"
        ]
        if 'best_hit_species' in row_ and row_['best_hit_species']:
            parts.append(f"Species: {row_['best_hit_species']}")
        if 'genus' in row_ and row_['genus']:
            parts.append(f"Genus: {row_['genus']}")
        if 'phylum' in row_ and row_['phylum']:
            parts.append(f"Phylum: {row_['phylum']}")
        if 'prevalence_pct' in row_:
            parts.append(f"Prevalence: {row_['prevalence_pct']:.1f}%")
        hover_text.append('<br>'.join(parts))
    fig = px.scatter(
        plot_data,
        x='complexity',
        y='importance_score',
        color='task',
        size='length',
        title='Sequence Complexity vs Feature Importance',
        labels={
            'complexity': 'Sequence Complexity (Shannon Entropy)', 
            'importance_score': 'Importance Score',
            'length': 'Length (bp)'
        }
    )
    fig.update_traces(hovertext=hover_text, hoverinfo='text')
    
    fig.update_layout(
        height=600,
        width=1000,
        template='plotly_white'
    )
    
    html_path = output_dir / 'sequence_complexity_vs_importance.html'
    png_path = output_dir / 'sequence_complexity_vs_importance.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=1000, height=600, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")
    
    # 3. Unitig length distribution per task (box + scatter, with BLAST hover)
    fig = go.Figure()
    for task in task_names:
        task_data = df_merged.filter(
            (pl.col('task') == task) & (pl.col('rank') <= top_k)
        )
        # Add box plot
        fig.add_trace(go.Box(
            y=task_data['length'].to_list(),
            name=task.replace('_', ' ').title(),
            boxmean='sd',
            marker=dict(opacity=0.5),
            showlegend=True
        ))
        # Add individual points with hover info
        hover_text = []
        for row in task_data.iter_rows(named=True):
            parts = [
                f"Feature: {row['feature_index']}",
                f"Length: {row['length']} bp",
                f"Rank: {row['rank']}"
            ]
            if 'best_hit_species' in row and row['best_hit_species']:
                parts.append(f"Species: {row['best_hit_species']}")
            if 'genus' in row and row['genus']:
                parts.append(f"Genus: {row['genus']}")
            if 'phylum' in row and row['phylum']:
                parts.append(f"Phylum: {row['phylum']}")
            if 'prevalence_pct' in row:
                parts.append(f"Prevalence: {row['prevalence_pct']:.1f}%")
            hover_text.append('<br>'.join(parts))
        fig.add_trace(go.Scatter(
            y=task_data['length'].to_list(),
            x=[task.replace('_', ' ').title()] * len(task_data),
            mode='markers',
            marker=dict(
                size=6,
                color=task_data['importance_score'].to_list(),
                colorscale='Viridis',
                showscale=(task == task_names[0]),
                colorbar=dict(title='Importance') if task == task_names[0] else None,
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
    fig.update_layout(
        title=f'Unitig Length Distribution for Top {top_k} Features',
        yaxis_title='Unitig Length (bp)',
        xaxis_title='Task',
        height=500,
        width=800,
        template='plotly_white'
    )
    html_path = output_dir / 'unitig_length_distribution.html'
    png_path = output_dir / 'unitig_length_distribution.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=800, height=500, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")
    
    # 4. Length vs GC content colored by importance
    # Build custom hover text
    hover_text = []
    for _, row_ in plot_data.iterrows():
        parts = [
            f"Feature: {row_['feature_index']}",
            f"Rank: {row_['rank']}",
            f"Length: {row_['length']} bp",
            f"GC: {row_['gc_content']:.1f}%",
            f"Importance: {row_['importance_score']:.4f}"
        ]
        if 'best_hit_species' in row_ and row_['best_hit_species']:
            parts.append(f"Species: {row_['best_hit_species']}")
        if 'genus' in row_ and row_['genus']:
            parts.append(f"Genus: {row_['genus']}")
        if 'phylum' in row_ and row_['phylum']:
            parts.append(f"Phylum: {row_['phylum']}")
        if 'prevalence_pct' in row_:
            parts.append(f"Prevalence: {row_['prevalence_pct']:.1f}%")
        hover_text.append('<br>'.join(parts))
    fig = px.scatter(
        plot_data,
        x='length',
        y='gc_content',
        color='importance_score',
        facet_col='task',
        facet_col_wrap=2,
        title='Unitig Length vs GC Content',
        labels={
            'length': 'Length (bp)',
            'gc_content': 'GC Content (%)',
            'importance_score': 'Importance'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_traces(hovertext=hover_text, hoverinfo='text')
    fig.update_layout(
        height=800,
        width=1400,
        template='plotly_white'
    )
    html_path = output_dir / 'length_vs_gc_by_task.html'
    png_path = output_dir / 'length_vs_gc_by_task.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=1400, height=800, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")


def create_feature_summary_table(
    df_merged: pl.DataFrame,
    task_names: List[str],
    output_dir: Path,
    top_k: int = 20
):
    """Create comprehensive summary table of top features with sequences."""
    logger.info("Creating feature summary tables...")
    
    # output_dir is paper/figures/feature_analysis
    # Need to go up 2 levels: parent.parent = paper/
    tables_dir = output_dir.parent.parent / 'tables' / 'feature_analysis'
    
    for task in task_names:
        task_data = df_merged.filter(pl.col('task') == task).sort('rank').head(top_k)
        
        # Base columns
        cols_to_include = [
            'rank',
            'feature_index',
            'id',
            'length',
            'gc_content',
            'complexity',
            'importance_score'
        ]
        
        # Add fraction statistics if available
        if 'prevalence_pct' in task_data.columns:
            cols_to_include.extend([
                'prevalence_pct',
                'n_samples_present',
                'mean_frac_present',
                'median_frac_present'
            ])
        
        # Add BLAST annotation if available
        if 'best_hit_species' in task_data.columns:
            cols_to_include.append('best_hit_species')
        
        # Select columns that exist
        available_cols = [col for col in cols_to_include if col in task_data.columns]
        summary = task_data.select(available_cols)
        
        # Save as CSV
        csv_path = tables_dir / f'top_features_{task}_with_sequences.csv'
        summary.write_csv(csv_path)
        logger.info(f"Saved {csv_path}")
        
        # Save as markdown
        md_path = tables_dir / f'top_features_{task}_with_sequences.md'
        with open(md_path, 'w') as f:
            f.write(f"# Top {top_k} Features for {task.replace('_', ' ').title()}\n\n")
            
            # Build header based on available columns
            has_frac = 'prevalence_pct' in task_data.columns
            has_blast = 'best_hit_species' in task_data.columns
            
            if has_frac and has_blast:
                f.write("| Rank | Feature | Unitig ID | Length | GC% | Complexity | Importance | Prevalence% | N Samples | Mean Frac | Median Frac | Species |\n")
                f.write("|------|---------|-----------|--------|-----|------------|------------|-------------|-----------|-----------|-------------|----------|\n")
            elif has_frac:
                f.write("| Rank | Feature | Unitig ID | Length | GC% | Complexity | Importance | Prevalence% | N Samples | Mean Frac | Median Frac |\n")
                f.write("|------|---------|-----------|--------|-----|------------|------------|-------------|-----------|-----------|-------------|\n")
            elif has_blast:
                f.write("| Rank | Feature Index | Unitig ID | Length (bp) | GC (%) | Complexity | Importance | Species |\n")
                f.write("|------|---------------|-----------|-------------|--------|------------|------------|----------|\n")
            else:
                f.write("| Rank | Feature Index | Unitig ID | Length (bp) | GC (%) | Complexity | Importance |\n")
                f.write("|------|---------------|-----------|-------------|--------|------------|------------|\n")
            
            for row in summary.iter_rows():
                if has_frac and has_blast:
                    species = row[11] if row[11] else 'No hit'
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.1f} | {row[5]:.2f} | {row[6]:.4f} | {row[7]:.1f} | {row[8]} | {row[9]:.3f} | {row[10]:.3f} | {species} |\n")
                elif has_frac:
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.1f} | {row[5]:.2f} | {row[6]:.4f} | {row[7]:.1f} | {row[8]} | {row[9]:.3f} | {row[10]:.3f} |\n")
                elif has_blast:
                    species = row[7] if row[7] else 'No hit'
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.2f} | {row[5]:.3f} | {row[6]:.6f} | {species} |\n")
                else:
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]:.2f} | {row[5]:.3f} | {row[6]:.6f} |\n")
        
        logger.info(f"Saved {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze unitig sequences for important features')
    parser.add_argument('--config', type=str, default='configs/feature_analysis.yaml',
                       help='Path to feature analysis config file')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Setup paths
    output_dir = Path(config['output']['figures_dir'])
    tables_dir = Path(config['output']['tables_dir'])
    
    # Get unitigs.fa path from matrix path
    matrix_path = Path(config['data']['matrix_path'])
    unitigs_fa = matrix_path.parent / 'unitigs.fa'
    
    if not unitigs_fa.exists():
        raise FileNotFoundError(f"unitigs.fa not found at {unitigs_fa}")
    
    # Load feature importance tables
    logger.info("Loading feature importance tables...")
    df_importance = load_feature_importance_tables(
        tables_dir,
        method='weight_based'  # Could make this configurable
    )
    
    task_names = df_importance['task'].unique().to_list()
    top_k = config['sequence_analysis']['top_k']
    
    # Get all top feature indices across all tasks
    top_features = df_importance.filter(
        pl.col('rank') <= top_k
    )['feature_index'].unique().to_list()
    
    logger.info(f"Analyzing {len(top_features)} unique features across {len(task_names)} tasks...")
    
    # Load fraction matrix
    frac_matrix_path = matrix_path.parent / 'unitigs.frac.mat'
    if frac_matrix_path.exists():
        logger.info(f"Loading fraction matrix from {frac_matrix_path}...")
        loader = MatrixLoader(str(frac_matrix_path))
        X_frac, sample_ids_frac, _ = loader.load()
        logger.info(f"Loaded fraction matrix: {X_frac.shape}")
        
        # Compute fraction statistics
        df_frac_stats = compute_fraction_statistics(X_frac, top_features)
    else:
        logger.warning(f"Fraction matrix not found at {frac_matrix_path}, skipping fraction statistics")
        df_frac_stats = None
    
    # Initialize UnitigAnalyzer
    logger.info("Loading unitig sequences...")
    analyzer = UnitigAnalyzer(str(unitigs_fa))
    
    # Compute statistics for all sequences (cached)
    all_stats = analyzer.compute_sequence_stats()
    
    # Filter to top features
    top_stats = all_stats.filter(pl.col('index').is_in(top_features))
    
    logger.info(f"Computed statistics for {len(top_stats)} sequences")
    
    # Merge with importance scores
    df_merged = df_importance.join(
        top_stats,
        left_on='feature_index',
        right_on='index',
        how='left'
    )
    
    # Merge fraction statistics if available
    if df_frac_stats is not None:
        df_merged = df_merged.join(
            df_frac_stats,
            on='feature_index',
            how='left'
        )
        logger.info("Merged fraction statistics with feature data")
    
    # Load and merge BLAST annotations
    df_blast = load_blast_annotations(tables_dir)
    if len(df_blast) > 0:
        # Merge on both feature_index and task
        df_merged = df_merged.join(
            df_blast.select(['feature_index', 'task', 'best_hit_species', 'phylum', 'family', 'genus']),
            on=['feature_index', 'task'],
            how='left'
        )
        logger.info("Merged BLAST annotations with feature data")
    
    # Ensure BLAST annotation columns are always present and filled
    for col in ["best_hit_species", "genus", "phylum"]:
        if col not in df_merged.columns:
            logger.warning(f"Column '{col}' missing from merged dataframe. Filling with 'No blast hit'.")
            df_merged = df_merged.with_columns(pl.lit('No blast hit').alias(col))
        else:
            n_missing = df_merged[col].is_null().sum() + (df_merged[col] == '').sum()
            if n_missing > 0:
                logger.warning(f"Column '{col}' has {n_missing} missing/empty values. Filling with 'No blast hit'.")
                df_merged = df_merged.with_columns(
                    pl.when((pl.col(col).is_null()) | (pl.col(col) == '')).then(pl.lit('No blast hit')).otherwise(pl.col(col)).alias(col)
                )
            # Warn if all values are 'No blast hit'
            n_no_blast = (df_merged[col] == 'No blast hit').sum()
            if n_no_blast == len(df_merged):
                logger.warning(f"Column '{col}' is 'No blast hit' for all features. BLAST annotation may be missing.")

    # Create visualizations
    create_sequence_analysis_plots(df_merged, task_names, output_dir, top_k)

    # Create summary tables
    create_feature_summary_table(df_merged, task_names, output_dir, top_k=top_k)

    logger.info("\n✅ Sequence analysis complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
