#!/usr/bin/env python3
"""
Generate BLAST summary table (sup_table_06)

Shows overall BLAST hit statistics for all features.

Input:
- results/feature_analysis/all_features_blast/blast_summary.json

Output:
- paper/tables/final/sup_table_06_blast_summary.tex

Process:
1. Load blast_summary.json
2. Extract overall stats (total_features, features_with_blast_hits, hit_rate_percent)
3. Extract identity distribution (pident_ranges)
4. Extract kingdom counts (taxonomy.kingdom_counts)
5. Extract unique_genera count
6. Format LaTeX table with 3 sections (Overall/Identity/Taxonomic)

All values come from JSON (no hardcoded values).
"""

import sys
import json
from pathlib import Path

# Add paper config
sys.path.insert(0, str(Path(__file__).parent))
from config import PATHS


def generate_blast_summary_table(blast_data: dict, output_path: Path) -> None:
    """Generate BLAST summary table from JSON data."""
    
    lines = []
    lines.append("\\centering")
    lines.append("\\caption{BLAST Hit Statistics for All Features}")
    lines.append("\\label{tab:blast_summary}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Metric} & \\textbf{Value} \\\\")
    lines.append("\\midrule")
    
    # Section 1: Overall Statistics
    lines.append("\\multicolumn{2}{l}{\\textit{Overall Statistics}} \\\\")
    lines.append(f"Total features analyzed & {blast_data['total_features']:,} \\\\")
    lines.append(f"Features with BLAST hits & {blast_data['features_with_blast_hits']:,} \\\\")
    lines.append(f"Overall hit rate & {blast_data['hit_rate_percent']:.2f}\\% \\\\")
    lines.append("\\midrule")
    
    # Section 2: Identity Distribution
    lines.append("\\multicolumn{2}{l}{\\textit{Identity Distribution}} \\\\")
    pident = blast_data['blast_statistics']['pident_ranges']
    lines.append(f"$\\geq$95\\% identity & {pident['>=95%']:,} \\\\")
    lines.append(f"90--95\\% identity & {pident['90-95%']:,} \\\\")
    lines.append(f"80--90\\% identity & {pident['80-90%']:,} \\\\")
    lines.append(f"$<$80\\% identity & {pident['<80%']:,} \\\\")
    lines.append("\\midrule")
    
    # Section 3: Taxonomic Distribution (Kingdom)
    lines.append("\\multicolumn{2}{l}{\\textit{Taxonomic Distribution (Kingdom)}} \\\\")
    kingdoms = blast_data['taxonomy']['kingdom_counts']
    
    # Order: Bacteria, Eukaryota, Archaea, Viruses, Unknown
    kingdom_order = ['Bacteria', 'Eukaryota', 'Archaea', 'Viruses', 'Unknown']
    for kingdom in kingdom_order:
        if kingdom in kingdoms:
            lines.append(f"{kingdom} & {kingdoms[kingdom]:,} \\\\")
    
    lines.append("\\midrule")
    
    # Unique genera
    unique_genera = blast_data['taxonomy']['unique_genera']
    lines.append(f"Unique genera identified & {unique_genera} \\\\")
    
    lines.append("\\botrule")
    lines.append("\\end{tabular}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 80)
    print("GENERATING BLAST SUMMARY TABLE (SUP TABLE 6)")
    print("=" * 80)
    print()
    
    # Step 1: Load BLAST summary JSON
    print("[1/2] Loading BLAST summary...")
    blast_json_path = Path("results/feature_analysis/all_features_blast/blast_summary.json")
    
    if not blast_json_path.exists():
        print(f"  ✗ ERROR: BLAST summary not found at {blast_json_path}")
        print("  Please run: sbatch scripts/feature_analysis/run_blast_all_features.sbatch")
        return
    
    with open(blast_json_path) as f:
        blast_data = json.load(f)
    
    print(f"  ✓ Loaded BLAST data:")
    print(f"    Total features: {blast_data['total_features']:,}")
    print(f"    Hit rate: {blast_data['hit_rate_percent']:.2f}%")
    print(f"    Unique genera: {blast_data['taxonomy']['unique_genera']}")
    print()
    
    # Step 2: Generate table
    print("[2/2] Generating LaTeX table...")
    output_path = Path(PATHS['tables_dir']) / "sup_table_06_blast_summary.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_blast_summary_table(blast_data, output_path)
    print(f"  ✓ {output_path.name}")
    print()
    
    print("=" * 80)
    print("✓ COMPLETE - BLAST summary table generated")
    print("=" * 80)


if __name__ == "__main__":
    main()
