#!/usr/bin/env python3
"""
Generate Matrix Generation Table (Supplementary Table 8)

PURPOSE:
    Document the feature matrix generation parameters and computational resources.

INPUTS:
    - data/matrices/large_matrix_3070_with_frac/muset_20251219_131306.log
      Contains muset run parameters and statistics
    - SLURM accounting data (job 60693849)

OUTPUTS:
    - paper/tables/final/sup_table_08_matrix_generation.tex

PROCESS:
    1. Parse muset log file for parameters
    2. Extract k-mer filtering statistics
    3. Extract unitig assembly statistics
    4. Format as LaTeX table with four categories:
       - Input (samples, data source)
       - k-mer Filtering (parameters and statistics)
       - Unitig Assembly (length, counts)
       - Computational (resources and runtime)

CONFIGURATION:
    All paths imported from config.py

HARDCODED VALUES:
    - Job ID: 60693849 (from SLURM accounting)
    - Node: maestro-2499 (from SLURM accounting)
    - Runtime: 3d 4h 45m (from SLURM accounting)
    - Memory: 500 GB (from SLURM accounting)
    - Training subset: 2,597 samples after filtering

DEPENDENCIES:
    - re, pathlib
    - config.py

USAGE:
    python scripts/paper/18_generate_matrix_generation_table.py
    
AUTHOR: Paper generation pipeline
"""

import sys
import re
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS


# ============================================================================
# LOG PARSING
# ============================================================================

def parse_muset_log(log_file):
    """Parse muset log file to extract parameters and statistics."""
    print(f"  Reading {log_file}")
    
    with open(log_file) as f:
        content = f.read()
    
    # Extract parameters
    params = {}
    
    # Basic parameters
    params['n_samples'] = re.search(r'(\d+) samples found', content).group(1)
    params['kmer_size'] = re.search(r'k-mer size \(-k\): (\d+)', content).group(1)
    params['min_abundance'] = re.search(r'minimum abundance \(-a\): (\d+)', content).group(1)
    params['min_length'] = re.search(r'minimum unitig length \(-l\): (\d+)', content).group(1)
    params['minimizer_size'] = re.search(r'minimizer size \(-m\): (\d+)', content).group(1)
    params['threads'] = re.search(r'threads \(-t\): (\d+)', content).group(1)
    
    # Thresholds - keep as separate parameters
    frac_absent = re.search(r'fraction of absent samples \(-f\): ([\d.]+)', content).group(1)
    frac_present = re.search(r'fraction of present samples \(-F\): ([\d.]+)', content).group(1)
    params['frac_absent'] = frac_absent
    params['frac_present'] = frac_present
    
    # Statistics - k-mers
    kmer_stats = re.search(r'(\d+)/(\d+) kmers retained', content)
    params['retained_kmers'] = kmer_stats.group(1)
    params['initial_kmers'] = kmer_stats.group(2)
    
    # Statistics - unitigs
    unitig_stats = re.search(r'(\d+)/(\d+) sequences retained', content)
    params['final_unitigs'] = unitig_stats.group(1)
    params['total_unitigs'] = unitig_stats.group(2)
    
    # Tool version
    params['muset_version'] = re.search(r'Running muset v([\d.]+)', content).group(1)
    
    # Check if logan unitigs
    params['is_logan'] = 'true' in re.search(r'input consists of logan unitigs \(--logan\): (\w+)', content).group(1)
    
    print(f"  ✓ Parsed {len(params)} parameters")
    
    return params


def format_large_number(num_str):
    """Format large number with commas for readability."""
    return f"{int(num_str):,}"


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_matrix_table(output_dir):
    """Generate table with matrix generation parameters."""
    print("\n[1/2] Parsing muset log file...")
    
    log_file = Path('data/matrices/large_matrix_3070_with_frac/muset_20251219_131306.log')
    
    if not log_file.exists():
        print(f"  ✗ ERROR: Log file not found: {log_file}")
        print(f"  Using hardcoded values instead...")
        # Fallback to hardcoded values
        params = {
            'n_samples': '3,070',
            'kmer_size': '31',
            'min_abundance': '2',
            'min_length': '61',
            'minimizer_size': '15',
            'threads': '32',
            'frac_present': '0.1',
            'frac_absent': '0.1',
            'retained_kmers': '18,953,119',
            'initial_kmers': '436,456,132,058',
            'final_unitigs': '107,480',
            'total_unitigs': '2,755,241',
            'muset_version': '0.6.1',
            'is_logan': True
        }
    else:
        params = parse_muset_log(log_file)
        # Format numbers with commas
        params['n_samples'] = format_large_number(params['n_samples'])
        params['retained_kmers'] = format_large_number(params['retained_kmers'])
        params['initial_kmers'] = format_large_number(params['initial_kmers'])
        params['final_unitigs'] = format_large_number(params['final_unitigs'])
        params['total_unitigs'] = format_large_number(params['total_unitigs'])
    
    print("\n[2/2] Generating LaTeX table...")
    
    data_source = "Logan unitigs" if params['is_logan'] else "FASTQ files"
    
    lines = []
    lines.append("\\centering")
    lines.append("\\caption{Feature matrix generation parameters and computational resources\\label{tab:matrix_generation}}")
    lines.append("\\addcontentsline{toc}{subsection}{\\protect\\numberline{\\thetable}Feature matrix generation parameters and computational resources}")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llr@{\\extracolsep{\\fill}}}")
    lines.append("\\toprule")
    lines.append("Category & Parameter & Value \\\\")
    lines.append("\\midrule")
    
    # Input section
    lines.append(f"\\multirow{{2}}{{*}}{{Input}} & Samples & {params['n_samples']} \\\\")
    lines.append(f" & Data source & {data_source} \\\\")
    lines.append("\\addlinespace")
    
    # k-mer Filtering section
    lines.append(f"\\multirow{{7}}{{*}}{{k-mer Filtering}} & k-mer size & {params['kmer_size']} \\\\")
    lines.append(f" & Minimum abundance & {params['min_abundance']} \\\\")
    lines.append(f" & Minimizer size & {params['minimizer_size']} \\\\")
    lines.append(f" & Min. fraction present (-F) & {params['frac_present']} \\\\")
    lines.append(f" & Max. fraction absent (-f) & {params['frac_absent']} \\\\")
    lines.append(f" & Initial k-mers & {params['initial_kmers']} \\\\")
    lines.append(f" & Retained k-mers & {params['retained_kmers']} \\\\")
    lines.append("\\addlinespace")
    
    # Unitig Assembly section
    lines.append(f"\\multirow{{3}}{{*}}{{Unitig Assembly}} & Minimum length & {params['min_length']} bp \\\\")
    lines.append(f" & Total unitigs assembled & {params['total_unitigs']} \\\\")
    lines.append(f" & Final unitigs & {params['final_unitigs']} \\\\")
    lines.append("\\addlinespace")
    
    # Computational section
    lines.append(f"\\multirow{{4}}{{*}}{{Computational}} & Tool & muset v{params['muset_version']} \\\\")
    lines.append(f" & CPU cores & {params['threads']} \\\\")
    lines.append(" & Memory & 500 GB \\\\")
    lines.append(" & Runtime & 3d 4h 45m \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\\\[2mm]")
    
    footnote_parts = [
        "Feature matrix constructed using muset on maestro cluster (node maestro-2499).",
        "k-mer matrix built with kmtricks, filtered by sample presence,",
        "assembled into unitigs with ggcat, and aggregated into mean coverage matrix.",
        f"Final matrix dimensions: {params['n_samples']} samples $\\times$ {params['final_unitigs']} unitigs.",
        "Training subset: 2,597 samples after filtering."
    ]
    lines.append("{\\footnotesize " + " ".join(footnote_parts) + "}")
    
    output_file = output_dir / "sup_table_08_matrix_generation.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  ✓ {output_file.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("GENERATING MATRIX GENERATION TABLE (SUP TABLE 8)")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(PATHS['tables_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate table
    generate_matrix_table(output_dir)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE - Matrix generation table generated")
    print("=" * 80)


if __name__ == '__main__':
    main()
