#!/usr/bin/env python3
"""
Generate Runtime and Memory Scalability Figure (Supplementary Figure 1)

PURPOSE:
    Analyze computational resource usage across all validation samples.
    Shows relationship between input FASTQ size and runtime/memory consumption.

INPUTS:
    - results/validation_predictions/*/. jobinfo: Job metadata with runtime (elapsed_seconds)
      and memory (memory_mb) for each validation sample
    - data/validation/raw/{sample}/*.fastq*: FASTQ input files for size calculation

OUTPUTS:
    - paper/figures/final/sup_01_runtime_memory.png
    - paper/figures/final/sup_01_runtime_memory.html

PROCESS:
    1. Scan all validation prediction directories for .jobinfo files
    2. Extract runtime (seconds → minutes) and memory (MB → GB)
    3. Calculate FASTQ input size in GB
    4. Filter to SUCCESS jobs only with complete data
    5. Create scatter plot:
        - X-axis: FASTQ file size (GB)
        - Y-axis: Runtime (minutes)
        - Color/marker: Memory allocation group
    6. Add linear regression line with R² value
    7. Report statistics and sample count

CONFIGURATION:
    All styling, paths, and constants imported from config.py:
    - PATHS: File locations for inputs/outputs
    - PLOT_CONFIG: Colors (Vivid palette), template, marker sizes

HARDCODED VALUES:
    - Minimum samples for plot: 10
    - Only SUCCESS jobs included
    - No fallback values - all data must be present

DEPENDENCIES:
    - pandas, numpy, plotly, scipy
    - config.py (same directory)

USAGE:
    python scripts/paper/generate_runtime_memory.py
    
AUTHOR: Generated via refactoring of 06_compare_predictions.py
"""

import sys
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# Add script directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS, PLOT_CONFIG


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

MIN_SAMPLES_FOR_PLOT = 10


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_runtime_memory_data() -> pd.DataFrame:
    """Collect runtime and memory data from validation predictions."""
    
    pred_dir = Path(PATHS.get('predictions_dir', 'results/validation_predictions'))
    
    print(f"Scanning {pred_dir} for job metadata...")
    
    data = []
    total_dirs = 0
    skipped_no_jobinfo = 0
    skipped_failed = 0
    skipped_incomplete = 0
    
    for sample_dir in pred_dir.glob("*"):
        if not sample_dir.is_dir():
            continue
        
        total_dirs += 1
        sample_id = sample_dir.name
        
        # Parse job info
        jobinfo = sample_dir / ".jobinfo"
        if not jobinfo.exists():
            skipped_no_jobinfo += 1
            continue
        
        try:
            with open(jobinfo) as f:
                info = json.load(f)
            
            # Skip if job didn't succeed
            if info.get('status') != 'SUCCESS':
                skipped_failed += 1
                continue
            
            # Extract memory (MB → GB) - NO DEFAULT
            memory_mb = info.get('memory_mb')
            if not memory_mb:
                skipped_incomplete += 1
                continue
            memory_gb = memory_mb / 1024.0
            
            # Extract runtime (seconds → minutes) - NO DEFAULT
            runtime_sec = info.get('elapsed_seconds')
            if not runtime_sec:
                skipped_incomplete += 1
                continue
            runtime_min = runtime_sec / 60.0
            
            # Calculate FASTQ size - NO FALLBACK
            fastq_dir = Path(f"data/validation/raw/{sample_id}")
            fastq_size_gb = 0.0
            
            if fastq_dir.exists():
                for fq in fastq_dir.glob("*.fastq*"):
                    fastq_size_gb += os.path.getsize(fq) / (1024**3)
            
            if fastq_size_gb == 0:
                skipped_incomplete += 1
                continue
            
            data.append({
                'sample_id': sample_id,
                'fastq_size_gb': fastq_size_gb,
                'runtime_min': runtime_min,
                'memory_gb': memory_gb
            })
            
        except Exception as e:
            skipped_incomplete += 1
            continue
    
    df = pd.DataFrame(data)
    
    print(f"\nData collection summary:")
    print(f"  Total directories: {total_dirs}")
    print(f"  Skipped (no .jobinfo): {skipped_no_jobinfo}")
    print(f"  Skipped (failed jobs): {skipped_failed}")
    print(f"  Skipped (incomplete data): {skipped_incomplete}")
    print(f"  ✓ Complete samples: {len(df)}")
    
    if len(df) < MIN_SAMPLES_FOR_PLOT:
        print(f"\n⚠ WARNING: Only {len(df)} samples with complete data (minimum: {MIN_SAMPLES_FOR_PLOT})")
    
    return df


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def generate_runtime_memory_figure(output_dir: Path) -> None:
    """Generate runtime and memory scalability figure."""
    
    # Collect data
    df = collect_runtime_memory_data()
    
    if len(df) < MIN_SAMPLES_FOR_PLOT:
        print(f"\n⚠ Insufficient data points ({len(df)} samples)")
        print(f"  Skipping sup_01_runtime_memory.png")
        return
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['fastq_size_gb'], 
        df['runtime_min']
    )
    r_squared = r_value ** 2
    
    print(f"\nLinear regression results:")
    print(f"  R² = {r_squared:.3f}")
    print(f"  Slope = {slope:.2f} min/GB")
    print(f"  P-value = {p_value:.2e}")
    
    # Create scatter plot with color by memory
    fig = go.Figure()
    
    # Group by memory allocation
    memory_groups = sorted(df['memory_gb'].unique())
    
    for idx, memory in enumerate(memory_groups):
        subset = df[df['memory_gb'] == memory]
        
        fig.add_trace(go.Scatter(
            x=subset['fastq_size_gb'],
            y=subset['runtime_min'],
            mode='markers',
            name=f'{memory:.0f} GB RAM',
            marker=dict(
                size=10,
                color=PLOT_CONFIG['colors']['palette'][idx % len(PLOT_CONFIG['colors']['palette'])],
                opacity=PLOT_CONFIG['marker_opacity'],
                line=dict(
                    color=PLOT_CONFIG['border_color'], 
                    width=1.5
                )
            )
        ))
    
    # Add regression line
    x_range = np.linspace(df['fastq_size_gb'].min(), df['fastq_size_gb'].max(), 100)
    y_fit = slope * x_range + intercept
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_fit,
        mode='lines',
        name=f'Linear fit (R² = {r_squared:.3f})',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'Runtime and Memory Scalability (n={len(df)} samples, R² = {r_squared:.3f})',
        xaxis_title='Input FASTQ Size (GB)',
        yaxis_title='Runtime (minutes)',
        template=PLOT_CONFIG['template'],
        height=600,
        width=900,
        font=dict(size=PLOT_CONFIG['font_size']),
        legend=dict(title='Memory Allocated')
    )
    
    # Save outputs
    output_file = output_dir / "sup_01_runtime_memory.png"
    fig.write_html(str(output_file.with_suffix('.html')))
    fig.write_image(str(output_file), width=900, height=600, scale=2)
    
    print(f"\n✓ {output_file.name}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  FASTQ size: {df['fastq_size_gb'].min():.2f} - {df['fastq_size_gb'].max():.2f} GB")
    print(f"  Runtime: {df['runtime_min'].min():.1f} - {df['runtime_min'].max():.1f} minutes")
    print(f"  Memory allocations: {memory_groups}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING RUNTIME/MEMORY SCALABILITY FIGURE")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figure
    print("\nGenerating runtime and memory scalability plot...")
    generate_runtime_memory_figure(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Runtime/memory figure generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
