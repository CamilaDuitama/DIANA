#!/usr/bin/env python3
"""
Generate runtime and memory scalability figure for diana-predict.

Creates a two-panel figure showing:
- Panel A: Runtime vs FASTQ file size
- Panel B: Memory usage vs FASTQ file size

Output: paper/figures/main_04_runtime_memory_scalability.png
"""

import json
import polars as pl
import plotly.graph_objects as go
from pathlib import Path
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plotly vivid color palette  
COLORS = {
    'scatter': '#636EFA',  # Blue
    'regression': '#EF553B'  # Red
}

def load_performance_data(predictions_dir: Path, metadata_file: Path):
    """Load runtime and memory data from .jobinfo files and metadata."""
    
    # Load metadata with FASTQ sizes
    metadata = pl.read_csv(metadata_file, separator='\t')
    
    # Parse fastq_bytes column (can have multiple files separated by ;)
    def parse_fastq_bytes(bytes_str):
        if not bytes_str or bytes_str == 'nan':
            return None
        try:
            # Sum all files if multiple
            sizes = [int(x) for x in str(bytes_str).split(';')]
            return sum(sizes)
        except:
            return None
    
    metadata = metadata.with_columns(
        pl.col('fastq_bytes').map_elements(parse_fastq_bytes, return_dtype=pl.Int64).alias('total_bytes')
    )
    
    # Load .jobinfo files
    jobinfo_files = list(predictions_dir.glob("*/.jobinfo"))
    logger.info(f"Found {len(jobinfo_files)} .jobinfo files")
    
    records = []
    for jobinfo_path in jobinfo_files:
        run_accession = jobinfo_path.parent.name
        
        try:
            with open(jobinfo_path) as f:
                jobinfo = json.load(f)
            
            # Get metadata for this sample
            sample_meta = metadata.filter(pl.col('run_accession') == run_accession)
            if len(sample_meta) == 0:
                continue
            
            total_bytes = sample_meta['total_bytes'][0]
            if total_bytes is None or total_bytes <= 0:
                continue
            
            records.append({
                'run_accession': run_accession,
                'fastq_size_gb': total_bytes / (1024**3),  # Convert to GB
                'runtime_minutes': jobinfo['elapsed_seconds'] / 60.0,
                'memory_gb': jobinfo['memory_mb'] / 1024.0,
                'status': jobinfo['status']
            })
        except Exception as e:
            logger.warning(f"Error processing {jobinfo_path}: {e}")
            continue
    
    df = pl.DataFrame(records)
    logger.info(f"Loaded {len(df)} samples with complete data")
    
    return df

def calculate_linear_fit(x, y):
    """Calculate linear regression and return slope, intercept, r-squared."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return slope, intercept, r_squared

def create_scalability_figure(df: pl.DataFrame, output_path: Path):
    """Create runtime scalability figure with memory indicated by color and size."""
    
    # Convert to pandas for plotly
    df_pd = df.to_pandas()
    
    # Calculate marker sizes based on memory allocation (scale for visibility)
    # Use sqrt scale for better visualization
    df_pd['marker_size'] = df_pd['memory_gb'].apply(lambda x: np.sqrt(max(x, 1)) * 3)
    
    # Calculate linear fit for runtime
    slope_runtime, intercept_runtime, r2_runtime = calculate_linear_fit(
        df_pd['fastq_size_gb'], df_pd['runtime_minutes']
    )
    
    # Create single figure
    fig = go.Figure()
    
    # Add scatter plot with color and size based on memory
    fig.add_trace(
        go.Scatter(
            x=df_pd['fastq_size_gb'],
            y=df_pd['runtime_minutes'],
            mode='markers',
            marker=dict(
                size=df_pd['marker_size'],
                color=df_pd['memory_gb'],
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=0.5, color='white'),
                sizemode='diameter',
                colorbar=dict(
                    title='Memory<br>Allocated (GB)',
                    thickness=20,
                    len=0.7,
                    x=1.02
                ),
                showscale=True
            ),
            name='Samples',
            showlegend=False,
            hovertemplate='FASTQ: %{x:.2f} GB<br>Runtime: %{y:.1f} min<br>Memory: %{marker.color:.0f} GB<extra></extra>'
        )
    )
    
    # Add regression line for runtime
    x_fit = np.linspace(df_pd['fastq_size_gb'].min(), df_pd['fastq_size_gb'].max(), 100)
    y_fit_runtime = slope_runtime * x_fit + intercept_runtime
    
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit_runtime,
            mode='lines',
            line=dict(color=COLORS['regression'], width=3, dash='dash'),
            name=f'Linear fit (R²={r2_runtime:.3f})',
            showlegend=True,
            hovertemplate=f'y = {slope_runtime:.2f}x + {intercept_runtime:.2f}<br>R² = {r2_runtime:.3f}<extra></extra>'
        )
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Input FASTQ Size (GB)", 
        title_font=dict(size=16),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    fig.update_yaxes(
        title_text="Runtime (minutes)", 
        title_font=dict(size=16),
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Performance Scalability of diana-predict',
            font=dict(size=20, color='black'),
            x=0.5,
            xanchor='center'
        ),
        height=600,
        width=900,
        font=dict(size=13),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=80, b=60, l=80, r=120),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='left',
            x=0.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), width=900, height=600, scale=2)
    
    # Log statistics
    logger.info(f"\nRuntime Scalability:")
    logger.info(f"  Slope: {slope_runtime:.2f} min/GB")
    logger.info(f"  Intercept: {intercept_runtime:.2f} min")
    logger.info(f"  R²: {r2_runtime:.3f}")
    logger.info(f"  Mean runtime: {df_pd['runtime_minutes'].mean():.1f} ± {df_pd['runtime_minutes'].std():.1f} min")
    logger.info(f"  Mean memory allocated: {df_pd['memory_gb'].mean():.1f} ± {df_pd['memory_gb'].std():.1f} GB")
    logger.info(f"  Memory range: {df_pd['memory_gb'].min():.0f} - {df_pd['memory_gb'].max():.0f} GB")
    
    file_size_kb = output_path.stat().st_size / 1024
    logger.info(f"\n✓ Saved: {output_path} ({file_size_kb:.0f} KB)")

def main():
    logger.info("Generating runtime and memory scalability figure...")
    
    predictions_dir = Path("results/validation_predictions")
    metadata_file = Path("paper/metadata/validation_metadata.tsv")
    output_path = Path("paper/figures/main_04_runtime_memory_scalability.png")
    
    # Load data
    df = load_performance_data(predictions_dir, metadata_file)
    
    if len(df) < 10:
        logger.error(f"Insufficient data: only {len(df)} samples found")
        return 1
    
    # Create figure
    create_scalability_figure(df, output_path)
    
    logger.info("\nDone!")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
