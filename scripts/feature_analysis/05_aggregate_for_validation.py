#!/usr/bin/env python3
"""
Aggregate Feature Analysis Results for Validation Plots
========================================================

Generates the TSV files expected by the validation comparison script:
- feature_importance_by_genus.tsv: Summarizes feature counts per genus per task
- blast_annotations.tsv: BLAST hit rates (copied from existing figure data)

This script consolidates the detailed outputs from scripts 01-03 into
the simple aggregated formats needed for main figure generation.

USAGE:
------
python scripts/feature_analysis/05_aggregate_for_validation.py
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def aggregate_feature_importance_by_genus():
    """Aggregate feature counts per genus from annotated feature tables.
    
    Uses intelligent taxonomy fallback hierarchy:
    - Primary: genus (if not Unknown/Uncultured/empty)
    - Fallback 1: family (if genus is uninformative)
    - Fallback 2: phylum (if family is also uninformative)
    - Fallback 3: "No BLAST hit" (only if best_hit_species is truly absent)
    
    Note: If there's ANY taxonomy information, even "Unknown", that means there 
    WAS a BLAST hit - just with incomplete annotation.
    """
    
    input_dir = Path("paper/tables/feature_analysis")
    output_dir = Path("results/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    all_data = []
    
    for task in tasks:
        csv_file = input_dir / f"top_features_{task}_annotated.csv"
        
        if not csv_file.exists():
            print(f"⚠ Warning: {csv_file} not found, skipping {task}")
            continue
        
        df = pd.read_csv(csv_file)
        
        # Create taxonomy label using intelligent hierarchy
        if 'best_hit_species' in df.columns and 'genus' in df.columns and 'family' in df.columns and 'phylum' in df.columns:
            # Check if there's a BLAST hit based on best_hit_species
            def get_best_taxonomy(row):
                # First check: is there a BLAST hit at all?
                has_blast_hit = (
                    pd.notna(row['best_hit_species']) and 
                    row['best_hit_species'] != '' and 
                    row['best_hit_species'] != 'No blast hit'
                )
                
                if not has_blast_hit:
                    return 'No BLAST hit'
                
                # If we have a BLAST hit, use the most specific taxonomy available
                # Try genus
                if pd.notna(row['genus']) and row['genus'] not in ['Unknown', 'Uncultured', '']:
                    return row['genus']
                # Try family
                if pd.notna(row['family']) and row['family'] not in ['Unknown', 'Uncultured', '']:
                    return row['family']
                # Try phylum
                if pd.notna(row['phylum']) and row['phylum'] not in ['Unknown', 'Uncultured', '']:
                    return row['phylum']
                # Has BLAST hit but all taxonomy is Unknown/Uncultured
                return 'Unknown taxonomy'
            
            df['taxonomy'] = df.apply(get_best_taxonomy, axis=1)
            
            taxonomy_counts = df['taxonomy'].value_counts().reset_index()
            taxonomy_counts.columns = ['genus', 'n_features']  # Keep 'genus' column name for compatibility
            taxonomy_counts['task'] = task
            all_data.append(taxonomy_counts)
        else:
            print(f"⚠ Warning: Missing taxonomy columns in {csv_file}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        output_file = output_dir / "feature_importance_by_genus.tsv"
        combined.to_csv(output_file, sep='\t', index=False)
        print(f"✓ Created {output_file}")
        print(f"  - {len(combined)} genus-task pairs")
        print(f"  - {combined['genus'].nunique()} unique taxonomic labels")
    else:
        print("✗ No data to aggregate")


def create_blast_annotations_summary():
    """Create BLAST annotations summary from existing BLAST results."""
    
    output_dir = Path("results/feature_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we have the BLAST hit rate figure data source
    blast_dir = Path("paper/blast_results")
    if blast_dir.exists():
        print(f"✓ BLAST results directory exists: {blast_dir}")
    
    # For now, create a simple summary from the annotated feature tables
    input_dir = Path("paper/tables/feature_analysis")
    tasks = ['sample_type', 'community_type', 'sample_host', 'material']
    
    all_annotations = []
    
    for task in tasks:
        csv_file = input_dir / f"top_features_{task}_annotated.csv"
        
        if not csv_file.exists():
            continue
        
        df = pd.read_csv(csv_file)
        
        # Extract BLAST hit information
        if 'best_hit_species' in df.columns:
            df['task'] = task
            
            # A valid BLAST hit is when we have a match to a sequence in the database
            # Incomplete taxonomy (Unknown, Uncultured) is still a valid hit
            df['has_blast_hit'] = (
                df['best_hit_species'].notna() & 
                (df['best_hit_species'] != '') &
                (df['best_hit_species'] != 'No blast hit')
            )
            
            all_annotations.append(df[[
                'task', 'feature_index', 'genus', 'best_hit_species', 
                'best_pident', 'best_evalue', 'has_blast_hit'
            ]])
    
    if all_annotations:
        combined = pd.concat(all_annotations, ignore_index=True)
        output_file = output_dir / "blast_annotations.tsv"
        combined.to_csv(output_file, sep='\t', index=False)
        print(f"✓ Created {output_file}")
        print(f"  - {len(combined)} annotated features")
        print(f"  - {combined['has_blast_hit'].sum()} with BLAST hits ({100*combined['has_blast_hit'].mean():.1f}%)")
    else:
        print("✗ No BLAST annotations found")


def create_blast_summary_table():
    """Create LaTeX table from BLAST summary JSON."""
    import json
    
    blast_summary_path = Path("results/feature_analysis/all_features_blast/blast_summary.json")
    output_dir = Path("paper/tables/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not blast_summary_path.exists():
        print(f"⚠ BLAST summary not found at {blast_summary_path}")
        return
    
    with open(blast_summary_path) as f:
        data = json.load(f)
    
    # Create LaTeX table
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{BLAST Hit Statistics for All Features}
\label{tab:blast_summary}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{l}{\textit{Overall Statistics}} \\
Total features analyzed & """ + f"{data['total_features']:,}" + r""" \\
Features with BLAST hits & """ + f"{data['features_with_blast_hits']:,}" + r""" \\
Overall hit rate & """ + f"{data['hit_rate_percent']:.2f}\%" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Identity Distribution}} \\
$\geq$95\% identity & """ + f"{data['blast_statistics']['pident_ranges']['>=95%']:,}" + r""" \\
90--95\% identity & """ + f"{data['blast_statistics']['pident_ranges']['90-95%']:,}" + r""" \\
80--90\% identity & """ + f"{data['blast_statistics']['pident_ranges']['80-90%']:,}" + r""" \\
$<$80\% identity & """ + f"{data['blast_statistics']['pident_ranges']['<80%']:,}" + r""" \\
\midrule
\multicolumn{2}{l}{\textit{Taxonomic Distribution (Kingdom)}} \\
Bacteria & """ + f"{data['taxonomy']['kingdom_counts']['Bacteria']:,}" + r""" \\
Eukaryota & """ + f"{data['taxonomy']['kingdom_counts']['Eukaryota']:,}" + r""" \\
Archaea & """ + f"{data['taxonomy']['kingdom_counts']['Archaea']:,}" + r""" \\
Viruses & """ + f"{data['taxonomy']['kingdom_counts']['Viruses']:,}" + r""" \\
Unknown & """ + f"{data['taxonomy']['kingdom_counts']['Unknown']:,}" + r""" \\
\midrule
Unique genera identified & """ + f"{data['taxonomy']['unique_genera']:,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    output_file = output_dir / "sup_table_06_blast_summary.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Created {output_file}")


def main():
    print("="*60)
    print("Aggregating Feature Analysis Results for Validation Plots")
    print("="*60)
    print()
    
    print("1. Aggregating feature importance by genus...")
    aggregate_feature_importance_by_genus()
    print()
    
    print("2. Creating BLAST annotations summary...")
    create_blast_annotations_summary()
    print()
    
    print("3. Creating BLAST summary table...")
    create_blast_summary_table()
    print()
    
    print("="*60)
    print("✓ Complete! Files saved to results/feature_analysis/ and paper/tables/final/")
    print("="*60)


if __name__ == "__main__":
    main()
