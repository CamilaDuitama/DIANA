#!/usr/bin/env python3
"""
Test script: Create unitig PCA plot for a validation sample.

Usage:
    python scripts/paper/test_unitig_pca.py --sample ERR10114862
"""

import sys
from pathlib import Path
import pickle
import polars as pl
import numpy as np
import argparse
import logging
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'configs'))
sys.path.insert(0, str(Path(__file__).parent))

from paper_config import PATHS

# Import from module with numeric prefix
import importlib.util
spec = importlib.util.spec_from_file_location(
    "project_pca", 
    Path(__file__).parent / "07_project_new_samples_pca.py"
)
project_pca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project_pca)
plot_unitig_pca_with_taxonomy = project_pca.plot_unitig_pca_with_taxonomy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test unitig PCA plotting')
    parser.add_argument('--sample', required=True, help='Sample ID (e.g., ERR10114862)')
    parser.add_argument('--taxonomy-level', default=None, 
                        choices=['kingdom', 'phylum', 'class', 'order', 'genus'],
                        help='Taxonomy level to color by (default: all 4 levels)')
    parser.add_argument('--top-n', type=int, default=10, 
                        help='Number of top taxa to show')
    args = parser.parse_args()
    
    sample_id = args.sample
    
    # If no specific level is provided, do all 4
    if args.taxonomy_level:
        taxonomy_levels = [args.taxonomy_level]
    else:
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'genus']
    
    print("="*80)
    print(f"Testing Unitig PCA Plot for {sample_id}")
    print("="*80)
    
    # 1. Load PCA reference (contains unitig PCA model and unitig IDs)
    pca_ref_path = Path('models/pca_reference.pkl')
    print(f"\n1. Loading PCA reference from {pca_ref_path}...")
    
    with open(pca_ref_path, 'rb') as f:
        pca_ref = pickle.load(f)
    
    # Get unitig-space PCA from the reference
    # NOTE: The sample PCA (pca_model) operates on samples (rows)
    # We need the unitig PCA which operates on features (columns)
    # This should be stored in the reference as 'unitig_pca' or computed from loadings
    
    if 'unitig_pca' in pca_ref:
        unitig_pca = pca_ref['unitig_pca']
        print(f"  ✓ Found unitig_pca in reference")
    else:
        # The PCA model's components_ ARE the unitig loadings
        # So we can use the main PCA model
        unitig_pca = pca_ref['pca_model']
        print(f"  ✓ Using pca_model components as unitig loadings")
    
    unitig_ids = np.array(pca_ref['unitig_ids'])
    print(f"  ✓ {len(unitig_ids)} unitigs in reference")
    
    # 2. Load sample's unitig fractions
    sample_dir = Path(f'results/validation_predictions/{sample_id}')
    fraction_file = sample_dir / f'{sample_id}_unitig_fraction.txt'
    
    if not fraction_file.exists():
        print(f"\n❌ ERROR: Fraction file not found: {fraction_file}")
        return 1
    
    print(f"\n2. Loading sample unitig fractions from {fraction_file}...")
    
    # Read unitig fractions (format: one fraction per line, aligned with reference unitig order)
    with open(fraction_file) as f:
        lines = f.readlines()
    
    sample_fractions = np.array([float(line.strip()) for line in lines if line.strip()])
    
    if len(sample_fractions) != len(unitig_ids):
        print(f"\n❌ ERROR: Fraction file has {len(sample_fractions)} values but reference has {len(unitig_ids)} unitigs")
        return 1
    
    n_present = (sample_fractions > 0).sum()
    print(f"  ✓ Loaded {len(sample_fractions)} unitig fractions")
    print(f"  ✓ {n_present}/{len(unitig_ids)} unitigs present in sample")
    
    # 3. Load BLAST annotations with taxids
    blast_path = Path('results/feature_analysis/all_features_blast/blast_results.txt')
    
    if not blast_path.exists():
        print(f"\n❌ ERROR: BLAST results not found: {blast_path}")
        print("   Run: scripts/feature_analysis/run_blast_all_features.sbatch first")
        return 1
    
    print(f"\n3. Loading BLAST annotations from {blast_path}...")
    
    # Load raw BLAST with taxids
    blast_df = pl.read_csv(
        blast_path,
        separator="\t",
        has_header=False,
        new_columns=[
            'unitig_id', 'subject_id', 'pident', 'length', 'mismatch', 'gapopen',
            'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'taxid', 'description'
        ],
        schema_overrides={'taxid': pl.Utf8, 'unitig_id': pl.Int64}
    )
    
    # Take best hit per unitig
    blast_df = (
        blast_df
        .sort('bitscore', descending=True)
        .group_by('unitig_id')
        .first()
    )
    
    print(f"  ✓ Loaded BLAST annotations for {len(blast_df)} unitigs")
    
    # 4. Create unitig PCA plots for each taxonomy level
    output_dir = Path('paper/figures/unitig_pca_test')
    
    print(f"\n4. Creating unitig PCA plots for {len(taxonomy_levels)} taxonomy levels...")
    print(f"   Levels: {', '.join(taxonomy_levels)}")
    print(f"   Output directory: {output_dir}")
    
    all_results = []
    
    for i, tax_level in enumerate(taxonomy_levels, 1):
        print(f"\n   [{i}/{len(taxonomy_levels)}] Processing {tax_level}...")
        
        result = plot_unitig_pca_with_taxonomy(
            sample_id=sample_id,
            sample_unitig_fraction=sample_fractions,
            ref_unitig_pca=unitig_pca,
            ref_unitig_ids=unitig_ids,
            blast_annotations=blast_df,
            taxonomy_level=tax_level,
            output_dir=output_dir,
            top_n=args.top_n
        )
        
        if result:
            all_results.append(result)
        else:
            print(f"   ⚠️  Failed to create {tax_level} plot")
    
    # Summary
    if all_results:
        print("\n" + "="*80)
        print("✓ SUCCESS!")
        print("="*80)
        print(f"\nCreated {len(all_results)} plots in: {output_dir}")
        print(f"Sample has {all_results[0]['n_sample_unitigs']} present unitigs\n")
        
        for result in all_results:
            print(f"\n{result['taxonomy_level'].upper()}:")
            print(f"  HTML: {result['html_path'].name}")
            if result['png_path']:
                print(f"  PNG:  {result['png_path'].name}")
            print(f"  Top {len(result['top_taxa'])} taxa: {', '.join(result['top_taxa'][:5])}" + 
                  (f" (+{len(result['top_taxa'])-5} more)" if len(result['top_taxa']) > 5 else ""))
        
        print(f"\nView plots in browser:")
        print(f"  file://{output_dir.absolute()}/")
        return 0
    else:
        print("\n❌ Failed to create any plots")
        return 1


if __name__ == '__main__':
    sys.exit(main())
