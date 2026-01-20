#!/usr/bin/env python3
"""
Generate supplementary figure showing memory allocation vs input data size.

Shows the relationship between FASTQ file size and memory required for 
successful prediction, including OOM retry escalation.

Output: paper/figures/supp_04_memory_vs_datasize.png
"""

import argparse
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import subprocess

# Plotly vivid color palette
PLOTLY_VIVID_COLORS = [
    '#636EFA',  # blue
    '#EF553B',  # red  
    '#00CC96',  # green
    '#AB63FA',  # purple
    '#FFA15A',  # orange
]


def get_fastq_size(sample_dir: Path) -> float:
    """Get total FASTQ file size in GB for a sample."""
    fastq_files = list(sample_dir.glob("*.fastq.gz")) + list(sample_dir.glob("*.fastq"))
    total_bytes = sum(f.stat().st_size for f in fastq_files if f.exists())
    return total_bytes / (1024**3)  # Convert to GB


def main():
    parser = argparse.ArgumentParser(description='Generate memory vs data size plot')
    parser.add_argument('--predictions-dir', type=str, default='results/validation_predictions',
                       help='Directory with prediction results')
    parser.add_argument('--fastq-dir', type=str, default='data/validation/raw',
                       help='Directory with FASTQ files')
    parser.add_argument('--output', type=str, default='paper/figures/supp_04_memory_vs_datasize.png',
                       help='Output figure path')
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    fastq_dir = Path(args.fastq_dir)
    output_path = Path(args.output)
    
    print("Collecting memory and data size information...")
    
    data = []
    for sample_dir in predictions_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        
        sample_id = sample_dir.name
        jobinfo_file = sample_dir / ".jobinfo"
        memory_history_file = sample_dir / ".memory_history"
        
        if not jobinfo_file.exists():
            continue
        
        with open(jobinfo_file) as f:
            jobinfo = json.load(f)
        
        # Get final memory allocation
        if memory_history_file.exists():
            with open(memory_history_file) as f:
                memory_mb = int(f.read().strip().split('\n')[-1])
        else:
            memory_mb = jobinfo.get('memory_mb', 0)
        
        memory_gb = memory_mb / 1024
        
        # Get FASTQ size
        sample_fastq_dir = fastq_dir / sample_id
        if not sample_fastq_dir.exists():
            continue
        
        fastq_size_gb = get_fastq_size(sample_fastq_dir)
        
        if fastq_size_gb == 0:
            continue
        
        status = jobinfo.get('status', 'UNKNOWN')
        
        data.append({
            'sample_id': sample_id,
            'fastq_size_gb': fastq_size_gb,
            'memory_gb': memory_gb,
            'status': status
        })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("⚠️  No data found. Check paths.")
        return
    
    print(f"Collected data for {len(df)} samples")
    print(f"  Successful: {len(df[df['status']=='SUCCESS'])}")
    print(f"  Failed: {len(df[df['status']!='SUCCESS'])}")
    
    # Define memory tiers for grouping
    memory_tiers = [32, 64, 128, 256, 512]
    
    def assign_memory_tier(mem_gb):
        """Assign samples to memory tier buckets"""
        for tier in memory_tiers:
            if mem_gb <= tier:
                return tier
        return memory_tiers[-1]
    
    df['memory_tier'] = df['memory_gb'].apply(assign_memory_tier)
    
    # Create boxplots - one for each memory tier
    fig = go.Figure()
    
    colors = {
        32: PLOTLY_VIVID_COLORS[0],   # blue
        64: PLOTLY_VIVID_COLORS[2],   # green
        128: PLOTLY_VIVID_COLORS[4],  # orange
        256: PLOTLY_VIVID_COLORS[3],  # purple
        512: PLOTLY_VIVID_COLORS[1]   # red
    }
    
    for tier in memory_tiers:
        tier_data = df[df['memory_tier'] == tier]
        if len(tier_data) == 0:
            continue
        
        fig.add_trace(go.Box(
            x=[f"{tier} GB"] * len(tier_data),
            y=tier_data['fastq_size_gb'],
            name=f"{tier} GB",
            marker_color=colors[tier],
            boxmean='sd'  # Show mean and standard deviation
        ))
    
    fig.update_layout(
        title='FASTQ File Size Distribution by Memory Allocation',
        xaxis_title='Memory Allocated (GB)',
        yaxis_title='FASTQ File Size (GB)',
        height=500,
        width=800,
        template='plotly_white',
        showlegend=False
    )
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path.with_suffix('.html'))
    fig.write_image(output_path, width=800, height=500, scale=2.5)  # 300 DPI
    
    print(f"✅ Generated supplementary figure: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics by memory tier
    print(f"\nSummary statistics by memory tier:")
    for tier in [32, 64, 128, 256, 512]:
        tier_data = df[df['memory_tier'] == tier]
        if len(tier_data) > 0:
            print(f"  {tier} GB: n={len(tier_data)}, "
                  f"FASTQ size: {tier_data['fastq_size_gb'].median():.2f} GB (median), "
                  f"range: {tier_data['fastq_size_gb'].min():.2f}-{tier_data['fastq_size_gb'].max():.2f} GB")


if __name__ == '__main__':
    main()
