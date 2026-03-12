#!/usr/bin/env python3
"""
Generate computational resources table (main_table_02)

Shows validation inference computational resources stratified by memory tier.

Input:
- results/validation_predictions/{sample}/.jobinfo (successful jobs)
- data/validation/raw/{sample}/*.fastq.gz (FASTQ file sizes)

Output:
- paper/tables/final/main_table_02_computational_resources.tex

Process:
1. Scan all .jobinfo files with status="SUCCESS"
2. Extract memory_mb, elapsed_seconds, cpus, FASTQ input size
3. Assign to memory tiers based on actual memory allocation
4. Group by tier, calculate N, mean±std runtime, mean±std input size
5. Format LaTeX table with tier stratification

Memory tiers from scripts/validation/submit_validation_with_retry.sh:
- MEMORY_TIERS=(32000 64000 128000 256000 512000)  # MB
- Maps to: 31.25GB, 62.5GB, 125GB, 250GB, 500GB (displayed as integers)
"""

import sys
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add paper config
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS


def assign_memory_tier(memory_mb: float) -> int:
    """
    Assign memory to standard tier.
    
    Memory tiers from submit_validation_with_retry.sh:
    32000, 64000, 128000, 256000, 512000 MB
    """
    # Convert MB to GB for display
    memory_gb = memory_mb / 1024.0
    
    # Round to nearest tier (in GB for display)
    if memory_mb <= 48000:  # < 48GB → 31GB tier
        return 31
    elif memory_mb <= 96000:  # < 96GB → 62GB tier
        return 62
    elif memory_mb <= 192000:  # < 192GB → 125GB tier
        return 125
    elif memory_mb <= 384000:  # < 384GB → 250GB tier
        return 250
    else:  # >= 384GB → 500GB tier
        return 500


def generate_computational_resources_table(output_path: Path) -> None:
    """Generate computational resources table stratified by memory tier."""
    
    # Collect data from validation predictions
    pred_dir = Path("results/validation_predictions")
    data = []
    
    print("  Scanning .jobinfo files...")
    
    for sample_dir in pred_dir.glob("*"):
        if not sample_dir.is_dir():
            continue
        
        sample_id = sample_dir.name
        jobinfo = sample_dir / ".jobinfo"
        
        if not jobinfo.exists():
            continue
        
        try:
            with open(jobinfo) as f:
                info = json.load(f)
            
            # Only include successful jobs
            if info.get('status') != 'SUCCESS':
                continue
            
            # Extract memory (MB)
            memory_mb = info.get('memory_mb', 0)
            if memory_mb == 0:
                continue
            
            # Extract runtime (seconds -> minutes)
            runtime_sec = info.get('elapsed_seconds', 0)
            runtime_min = runtime_sec / 60.0 if runtime_sec else 0
            if runtime_min == 0:
                continue
            
            # Extract CPUs from jobinfo (not hardcoded)
            cpus = info.get('cpus', 0)
            if cpus == 0:
                continue
            
            # Get FASTQ input size
            fastq_dir = Path(f"data/validation/raw/{sample_id}")
            fastq_size_gb = 0.0
            if fastq_dir.exists():
                for fq in fastq_dir.glob("*.fastq*"):
                    fastq_size_gb += os.path.getsize(fq) / (1024**3)
            
            data.append({
                'memory_mb': memory_mb,
                'runtime_min': runtime_min,
                'input_size_gb': fastq_size_gb,
                'cpus': cpus
            })
        except Exception as e:
            continue
    
    if len(data) == 0:
        print(f"  ⚠ No computational resource data found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    total_samples = len(df)
    print(f"  ✓ Loaded data for {total_samples} successful samples")
    
    # Assign memory tiers
    df['tier'] = df['memory_mb'].apply(assign_memory_tier)
    
    # Aggregate by tier
    tier_stats = df.groupby('tier').agg({
        'runtime_min': ['count', 'mean', 'std'],
        'input_size_gb': ['mean', 'std'],
        'cpus': 'first'  # CPUs should be constant per tier
    }).reset_index()
    
    tier_stats.columns = ['tier', 'n', 'runtime_mean', 'runtime_std', 'input_mean', 'input_std', 'cpus']
    tier_stats = tier_stats.sort_values('tier')
    
    # Generate LaTeX table
    lines = []
    lines.append("\\begin{table}")
    lines.append("\\centering")
    lines.append(f"\\caption{{Validation inference computational resources stratified by memory tier ({total_samples} samples)}}")
    lines.append("\\label{tab:resources}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Memory (GB) & CPUs & N & Runtime (min) & Input size (GB) \\\\")
    lines.append("\\midrule")
    
    for _, row in tier_stats.iterrows():
        runtime_str = f"{row['runtime_mean']:.2f} $\\pm$ {row['runtime_std']:.2f}" if not np.isnan(row['runtime_std']) else f"{row['runtime_mean']:.2f}"
        input_str = f"{row['input_mean']:.2f} $\\pm$ {row['input_std']:.2f}" if not np.isnan(row['input_std']) else f"{row['input_mean']:.2f}"
        lines.append(f"{int(row['tier'])} & {int(row['cpus'])} & {int(row['n'])} & {runtime_str} & {input_str} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("{\\footnotesize Runtime includes the creation of a vector of unitig feature abundances per sample and neural network inference.}")
    lines.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 80)
    print("GENERATING COMPUTATIONAL RESOURCES TABLE (MAIN TABLE 2)")
    print("=" * 80)
    print()
    
    # Step 1: Generate table
    print("[1/2] Collecting computational resource data...")
    output_path = Path(PATHS['tables_dir']) / "main_table_02_computational_resources.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_computational_resources_table(output_path)
    print()
    
    # Step 2: Report
    print("[2/2] Summary:")
    print(f"  ✓ {output_path.name}")
    print()
    
    print("=" * 80)
    print("✓ COMPLETE - Computational resources table generated")
    print("=" * 80)


if __name__ == "__main__":
    main()
