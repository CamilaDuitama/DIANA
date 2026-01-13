#!/usr/bin/env python3
"""
Prepare Combined Validation Metadata (Host-Associated + Environmental)
========================================================================

Combines host-associated and environmental samples from AncientMetagenomeDir v25.09.0
into a single metadata file ready for ENA expansion.

USAGE:
------
python scripts/validation/00_prepare_combined_metadata.py \
    --host-associated data/validation/ancientmetagenome-hostassociated_samples_v25.09.0.tsv \
    --environmental data/validation/ancientmetagenome-environmental_samples_v25.09.0.tsv \
    --output data/validation/validation_metadata_v25.09.0_COMBINED.tsv

This creates a unified metadata file with:
- All host-associated samples (sample_source='host_associated')
- All environmental samples (sample_source='environmental')
- Standardized columns for both types
- Ready for 01_expand_metadata.py
"""

import argparse
import logging
from pathlib import Path
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_host_associated_metadata(file_path: Path) -> pl.DataFrame:
    """
    Load and standardize host-associated samples metadata.
    
    Returns:
        DataFrame with standardized columns
    """
    logger.info(f"Loading host-associated samples from {file_path}")
    
    df = pl.read_csv(file_path, separator='\t', null_values=['NA', ''])
    
    logger.info(f"  Loaded {len(df)} host-associated samples")
    
    # Select and rename columns to match expected format
    # AncientMetagenomeDir host-associated has these columns already
    standardized = df.select([
        pl.col('archive_accession'),
        pl.col('sample_name'),
        pl.lit('ancient_metagenome').alias('sample_type'),
        pl.lit('host_associated').alias('sample_source'),
        pl.col('sample_host'),
        pl.col('material'),
        pl.col('community_type'),
        pl.col('geo_loc_name'),
        pl.col('site_name'),
        pl.col('latitude'),
        pl.col('longitude'),
        pl.col('sample_age'),
        pl.col('sample_age_doi'),
        pl.col('project_name'),
        pl.col('publication_year'),
        pl.col('publication_doi'),
    ])
    
    return standardized


def prepare_environmental_metadata(file_path: Path) -> pl.DataFrame:
    """
    Load and standardize environmental samples metadata.
    
    Environmental samples need to be mapped to match host-associated format:
    - sample_source: 'environmental'
    - sample_host: 'Not applicable - env sample'
    - community_type: 'Not applicable - env sample'
    
    Returns:
        DataFrame with standardized columns matching host-associated format
    """
    logger.info(f"Loading environmental samples from {file_path}")
    
    df = pl.read_csv(file_path, separator='\t', null_values=['NA', ''])
    
    logger.info(f"  Loaded {len(df)} environmental samples")
    
    # Environmental samples have different columns, need to map them
    standardized = df.select([
        pl.col('archive_accession'),
        pl.col('sample_name'),
        pl.lit('ancient_metagenome').alias('sample_type'),
        pl.lit('environmental').alias('sample_source'),
        pl.lit('Not applicable - env sample').alias('sample_host'),
        pl.col('material'),
        pl.lit('Not applicable - env sample').alias('community_type'),
        pl.col('geo_loc_name'),
        pl.col('site_name'),
        pl.col('latitude'),
        pl.col('longitude'),
        pl.col('sample_age'),
        pl.col('sample_age_doi'),
        pl.col('project_name'),
        pl.col('publication_year'),
        pl.col('publication_doi'),
    ])
    
    return standardized


def combine_metadata(
    host_file: Path,
    env_file: Path,
    output_file: Path
) -> pl.DataFrame:
    """
    Combine host-associated and environmental metadata into single file.
    
    Args:
        host_file: Path to host-associated TSV
        env_file: Path to environmental TSV
        output_file: Path for combined output TSV
        
    Returns:
        Combined DataFrame
    """
    # Load and standardize both datasets
    host_df = prepare_host_associated_metadata(host_file)
    env_df = prepare_environmental_metadata(env_file)
    
    # Combine
    logger.info("Combining datasets...")
    combined = pl.concat([host_df, env_df], how='vertical')
    
    logger.info(f"  Combined total: {len(combined)} samples")
    logger.info(f"    Host-associated: {len(host_df)} samples")
    logger.info(f"    Environmental: {len(env_df)} samples")
    
    # Summary statistics
    logger.info("\nSample source distribution:")
    source_counts = combined.group_by('sample_source').len().sort('len', descending=True)
    for row in source_counts.iter_rows(named=True):
        logger.info(f"  {row['sample_source']}: {row['len']}")
    
    logger.info("\nMaterial distribution:")
    material_counts = combined.group_by('material').len().sort('len', descending=True)
    for row in material_counts.head(15).iter_rows(named=True):
        logger.info(f"  {row['material']}: {row['len']}")
    
    # Save combined metadata
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output_file, separator='\t')
    logger.info(f"\n✓ Saved combined metadata to {output_file}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Prepare combined validation metadata (host-associated + environmental)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--host-associated',
        type=Path,
        required=True,
        help='Host-associated samples TSV from AncientMetagenomeDir'
    )
    parser.add_argument(
        '--environmental',
        type=Path,
        required=True,
        help='Environmental samples TSV from AncientMetagenomeDir'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output combined metadata TSV'
    )
    
    args = parser.parse_args()
    
    # Combine metadata
    combined_df = combine_metadata(
        args.host_associated,
        args.environmental,
        args.output
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Unique sample names: {combined_df['sample_name'].n_unique()}")
    logger.info(f"Unique archive accessions: {combined_df['archive_accession'].n_unique()}")
    logger.info("\nNext step:")
    logger.info("  python scripts/validation/01_expand_metadata.py \\")
    logger.info(f"    --input {args.output} \\")
    logger.info("    --output data/validation/validation_metadata_expanded_RAW.tsv \\")
    logger.info("    --cache data/validation/ena_cache.json")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
