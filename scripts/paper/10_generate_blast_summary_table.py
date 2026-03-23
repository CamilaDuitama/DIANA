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
    lines.append("\\caption{BLAST annotation of unitig features against the NCBI nt database\\label{tab:blast_summary}}")
    lines.append("\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lr}")
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
    lines.append("\\multicolumn{2}{l}{\\textit{Identity Distribution (best hit per feature)}} \\\\")
    pident = blast_data['blast_statistics']['pident_ranges']
    lines.append(f"$\\geq$95\\% identity & {pident['>=95%']:,} \\\\")
    lines.append(f"90--95\\% identity & {pident['90-95%']:,} \\\\")
    lines.append(f"80--90\\% identity & {pident['80-90%']:,} \\\\")
    lines.append(f"$<$80\\% identity & {pident['<80%']:,} \\\\")
    lines.append("\\midrule")

    # Section 3: Top 10 species by number of features
    lines.append("\\multicolumn{2}{l}{\\textit{Top 10 most frequent species (best informative hit per feature)}} \\\\")
    top_species = blast_data['taxonomy']['top_species_by_feature_count']
    for species, count in list(top_species.items())[:10]:
        lines.append(f"\\textit{{{species}}} & {count:,} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\addcontentsline{toc}{subsection}{Supplementary Table 6: BLAST hit statistics}")
    lines.append("\\\\[2mm]")
    lines.append(
        "{\\footnotesize "
        "\\textbf{Overall Statistics} and \\textbf{Identity Distribution}: for each of the %s unitig features, "
        "the single best BLAST hit (highest bitscore) against the NCBI nt database is used, "
        "regardless of taxonomic content (megaBLAST, E-value $\\leq 10^{-5}$). "
        "\\textbf{Species ranking}: to assign a biologically meaningful species to each feature, "
        "hits whose description begins with a non-specific term "
        "(\\textit{uncultured}, \\textit{unclassified}, \\textit{metagenome}, "
        "\\textit{environmental sample}, \\textit{synthetic}, etc.) are excluded; "
        "the best remaining (informative) hit is used instead. "
        "Species names are taken from the first two words of the hit description. "
        "Features with no informative hit are omitted from the species ranking "
        "but are still counted in the overall hit rate.}" %
        f"{blast_data['total_features']:,}"
    )

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
