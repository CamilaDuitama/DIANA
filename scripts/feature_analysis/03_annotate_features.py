#!/usr/bin/env python3
"""
Annotate Features - BLAST-based Taxonomic Annotation of Important Unitigs
==========================================================================

Takes the top important unitig sequences and performs BLAST analysis to:
- Identify taxonomic origin (species/genus/family)
- Annotate features with best BLAST hits
- Create taxonomic breakdown visualizations
- Generate enhanced summary tables with taxonomy

REQUIRES:
---------
- BLAST+ (load via: module load blast)
- BLAST database (e.g., /local/databases/index/blast+/nt)
- Top features with sequences from 02_analyze_feature_sequences.py

OUTPUT:
-------
- BLAST results (tabular format)
- Enhanced tables with taxonomy annotations
- Taxonomic composition plots per task
- Sunburst/bar charts showing predictive taxa

USAGE:
------
# First ensure BLAST is loaded
module load blast

# Run annotation
python scripts/feature_analysis/03_annotate_features.py \\
    --config configs/feature_analysis.yaml \\
    --blast-db /local/databases/index/blast+/nt \\
    --top-k 20
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import tempfile
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import logging
from collections import Counter, defaultdict

try:
    from ete3 import NCBITaxa
    HAS_ETE3 = True
except ImportError:
    HAS_ETE3 = False
    logging.warning("ete3 not available. Falling back to simple genus extraction.")

from diana.data.unitig_analyzer import UnitigAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_feature_tables_with_sequences(tables_dir: Path, task_names: List[str], top_k: int = 20) -> pl.DataFrame:
    """Load all feature tables with sequence info and filter to top_k per task."""
    all_data = []
    
    for task in task_names:
        csv_path = tables_dir / f'top_features_{task}_with_sequences.csv'
        if not csv_path.exists():
            logger.warning(f"Table not found: {csv_path}")
            continue
        
        df = pl.read_csv(csv_path)
        # Filter to top_k features for this task (sorted by rank)
        if 'rank' in df.columns:
            df = df.sort('rank').head(top_k)
        else:
            df = df.head(top_k)
        
        df = df.with_columns(pl.lit(task).alias('task'))
        all_data.append(df)
        logger.info(f"Loaded {len(df)} features for task '{task}' (top_k={top_k})")
    
    if not all_data:
        raise FileNotFoundError("No feature tables found")
    
    combined = pl.concat(all_data)
    logger.info(f"Total: {len(combined)} features from {len(task_names)} tasks")
    return combined


def extract_sequences_for_blast(
    df_features: pl.DataFrame,
    analyzer: UnitigAnalyzer,
    output_fasta: Path
) -> Dict[str, int]:
    """
    Extract sequences for BLAST and write to FASTA.
    Returns mapping of unitig_id -> feature_index.
    """
    logger.info("Extracting sequences for BLAST...")
    
    # Get unique feature indices
    unique_features = df_features.select(['feature_index', 'id']).unique()
    feature_indices = unique_features['feature_index'].to_list()
    unitig_ids = unique_features['id'].to_list()
    
    # Get sequences by matrix indices (not FASTA IDs)
    sequences_dict = analyzer.get_sequences_by_indices(feature_indices)
    
    # Create mapping from feature index to FASTA ID (O(1) lookup)
    idx_to_id = dict(zip(feature_indices, unitig_ids))
    id_to_index = dict(zip(unitig_ids, feature_indices))
    
    # Write to FASTA using unitig IDs from the table as headers
    with open(output_fasta, 'w') as f:
        for feat_idx, seq in sequences_dict.items():
            # O(1) lookup instead of O(n) list.index()
            unitig_id = idx_to_id[feat_idx]
            f.write(f">{unitig_id}\n{seq}\n")
    
    logger.info(f"Wrote {len(sequences_dict)} sequences to {output_fasta}")
    return id_to_index


def run_blast(
    query_fasta: Path,
    blast_db: str,
    output_file: Path,
    num_threads: int = 4,
    max_target_seqs: int = 5,
    evalue: float = 1e-5
) -> bool:
    """
    Run BLAST analysis on query sequences.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"Running BLAST against {blast_db}...")
    
    # BLAST command with tabular output format 6
    # Using newer BLAST version (2.17.0) which handles gi fields better
    # Columns: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids stitle
    cmd = [
        'blastn',
        '-query', str(query_fasta),
        '-db', blast_db,
        '-out', str(output_file),
        '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids stitle',
        '-num_threads', str(num_threads),
        '-max_target_seqs', str(max_target_seqs),
        '-evalue', str(evalue),
        '-task', 'blastn'
    ]
    
    logger.info(f"BLAST command: {' '.join(cmd)}")
    logger.info(f"Output will be written to: {output_file}")
    logger.info(f"Number of threads: {num_threads}")
    logger.info("\n*** BLAST is running - progress will be shown below ***\n")
    
    try:
        # Don't capture output so user can see BLAST progress in real-time
        result = subprocess.run(
            cmd,
            check=True,
            timeout=7200  # 2 hour timeout (nt database can be slow)
        )
        
        logger.info("\n*** BLAST completed successfully ***")
        
        # Check output file was created and has content
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"BLAST results written: {output_file.stat().st_size:,} bytes")
        else:
            logger.warning("BLAST output file is empty - no hits found")
        
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"BLAST failed with exit code {e.returncode}")
        logger.error(f"stderr: {e.stderr}")
        return False
    
    except subprocess.TimeoutExpired:
        logger.error("BLAST timed out after 2 hours")
        logger.error("You can resume by running again with partial results")
        return False
    
    except FileNotFoundError:
        logger.error("blastn not found. Make sure BLAST+ is installed and loaded (module load blast)")
        return False


def select_best_hit(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    Select the best BLAST hit from a group.
    
    Primary criterion: Highest bitscore
    Tie-breaker: Lowest E-value
    
    This ensures a scientifically robust and reproducible choice.
    """
    return group_df.sort(
        by=['bitscore', 'evalue'],
        descending=[True, False]  # Max bitscore, min evalue
    ).head(1)


def parse_blast_results(blast_output: Path) -> pl.DataFrame:
    """
    Parse BLAST tabular output and extract best hit per query.
    
    Returns DataFrame with columns:
    - query_id: unitig ID
    - best_hit_species: scientific name of best hit
    - best_hit_title: full description
    - pident: percent identity
    - length: alignment length
    - evalue: E-value
    - bitscore: bit score
    - taxid: taxonomy ID
    """
    logger.info(f"Parsing BLAST results from {blast_output}...")
    
    if not blast_output.exists():
        logger.error(f"BLAST output file not found: {blast_output}")
        return pl.DataFrame()
    
    # Check if file is empty
    if blast_output.stat().st_size == 0:
        logger.warning("BLAST output is empty - no hits found")
        return pl.DataFrame()
    
    # Read BLAST output
    # Format: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids stitle
    column_names = [
        'query_id', 'subject_id', 'pident', 'length', 'mismatch', 'gapopen',
        'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore',
        'taxid', 'title'
    ]
    
    # Specify schema to handle numeric types correctly
    schema_overrides = {
        'query_id': pl.Int64,
        'pident': pl.Float64,
        'length': pl.Int64,
        'mismatch': pl.Int64,
        'gapopen': pl.Int64,
        'qstart': pl.Int64,
        'qend': pl.Int64,
        'sstart': pl.Int64,
        'send': pl.Int64,
        'evalue': pl.Float64,
        'bitscore': pl.Float64,  # Can have decimal values
        'taxid': pl.Utf8,
        'title': pl.Utf8
    }
    
    try:
        df = pl.read_csv(
            blast_output,
            separator='\t',
            has_header=False,
            new_columns=column_names,
            schema_overrides=schema_overrides,
            truncate_ragged_lines=True
        )
        
        logger.info(f"Loaded {len(df)} BLAST hits")
        
        # Extract species from title (format: "Genus species strain/description...")
        # Take first two words as genus + species
        df = df.with_columns([
            pl.col('title').str.extract(r'^(\S+\s+\S+)', 1).alias('species_from_title')
        ])
        
        # Get best hit per query using vectorized sort + group_by + head
        # Sort by bitscore (desc) and evalue (asc), then take first hit per query
        best_hits_df = df.sort(
            by=['bitscore', 'evalue'],
            descending=[True, False]
        ).group_by('query_id', maintain_order=True).head(1)
        
        # Rename columns
        best_hits = best_hits_df.select([
            'query_id',
            pl.col('species_from_title').alias('best_hit_species'),
            pl.col('title').alias('best_hit_title'),
            pl.col('pident').alias('best_pident'),
            pl.col('length').alias('best_length'),
            pl.col('evalue').alias('best_evalue'),
            pl.col('bitscore').alias('best_bitscore'),
            pl.col('taxid').alias('best_taxid')
        ])
        
        # Add hit count
        hit_counts = df.group_by('query_id').agg([
            pl.count().alias('num_hits')
        ])
        best_hits = best_hits.join(hit_counts, on='query_id', how='left')
        
        logger.info(f"Found best hits for {len(best_hits)} queries")
        return best_hits
    
    except Exception as e:
        logger.error(f"Error parsing BLAST results: {e}")
        return pl.DataFrame()


class TaxonomicResolver:
    """
    Resolves taxonomy IDs to full lineage using NCBI taxonomy database.
    Uses ete3 if available, otherwise falls back to simple genus extraction.
    """
    
    def __init__(self):
        self.ncbi = None
        if HAS_ETE3:
            try:
                self.ncbi = NCBITaxa()
                # Update database if needed (this might take time on first run)
                logger.info("Loaded NCBI taxonomy database")
            except Exception as e:
                logger.warning(f"Could not initialize NCBI taxonomy database: {e}")
                self.ncbi = None
    
    def get_lineage(self, taxid: Optional[int], species_name: str = '') -> Dict[str, str]:
        """
        Get full taxonomic lineage from taxonomy ID.
        Falls back to species name parsing if taxid lookup fails.
        """
        # Try ete3 first if available
        if self.ncbi and taxid:
            try:
                taxid_int = int(taxid)
                lineage = self.ncbi.get_lineage(taxid_int)
                
                if lineage:
                    # Get rank names
                    names = self.ncbi.get_taxid_translator(lineage)
                    ranks = self.ncbi.get_rank(lineage)
                    
                    # Extract specific ranks
                    result = {
                        'kingdom': 'Unknown',
                        'phylum': 'Unknown',
                        'class': 'Unknown',
                        'order': 'Unknown',
                        'family': 'Unknown',
                        'genus': 'Unknown',
                        'species': 'Unknown'
                    }
                    
                    for tid in lineage:
                        rank = ranks.get(tid, '')
                        name = names.get(tid, 'Unknown')
                        
                        if rank in result:
                            result[rank] = name
                    
                    return result
            
            except Exception as e:
                logger.debug(f"Could not resolve taxid {taxid}: {e}")
        
        # Fallback to simple genus extraction from species name
        return self._extract_from_species_name(species_name)
    
    def get_lineages_batch(self, taxid_species_pairs: List[Tuple[Optional[int], str]]) -> List[Dict[str, str]]:
        """
        Batch process multiple taxids to reduce database query overhead.
        Returns list of lineage dictionaries in the same order as input.
        """
        if not self.ncbi or not taxid_species_pairs:
            # Fallback to individual processing
            return [self.get_lineage(tid, sp) for tid, sp in taxid_species_pairs]
        
        results = []
        
        # Collect all valid taxids
        valid_taxids = []
        taxid_to_idx = {}  # Map taxid to its positions in the input
        
        for idx, (taxid, species) in enumerate(taxid_species_pairs):
            if taxid:
                try:
                    tid_int = int(taxid)
                    if tid_int not in taxid_to_idx:
                        valid_taxids.append(tid_int)
                        taxid_to_idx[tid_int] = []
                    taxid_to_idx[tid_int].append(idx)
                except (ValueError, TypeError):
                    pass
        
        # Batch fetch lineages for all unique taxids
        lineage_cache = {}
        if valid_taxids:
            try:
                # Get all lineages at once
                for tid in valid_taxids:
                    lineage = self.ncbi.get_lineage(tid)
                    if lineage:
                        lineage_cache[tid] = lineage
                
                # Batch fetch names and ranks
                all_tids = set()
                for lineage in lineage_cache.values():
                    all_tids.update(lineage)
                
                if all_tids:
                    names_dict = self.ncbi.get_taxid_translator(list(all_tids))
                    ranks_dict = self.ncbi.get_rank(list(all_tids))
                else:
                    names_dict = {}
                    ranks_dict = {}
                
                # Build results for each taxid
                for tid in valid_taxids:
                    if tid in lineage_cache:
                        lineage = lineage_cache[tid]
                        result = {
                            'kingdom': 'Unknown',
                            'phylum': 'Unknown',
                            'class': 'Unknown',
                            'order': 'Unknown',
                            'family': 'Unknown',
                            'genus': 'Unknown',
                            'species': 'Unknown'
                        }
                        
                        for t in lineage:
                            rank = ranks_dict.get(t, '')
                            name = names_dict.get(t, 'Unknown')
                            if rank in result:
                                result[rank] = name
                        
                        lineage_cache[tid] = result  # Replace with processed result
            
            except Exception as e:
                logger.debug(f"Batch lineage lookup failed: {e}")
        
        # Build final results in input order
        for idx, (taxid, species) in enumerate(taxid_species_pairs):
            if taxid:
                try:
                    tid_int = int(taxid)
                    if tid_int in lineage_cache:
                        results.append(lineage_cache[tid_int])
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Fallback
            results.append(self._extract_from_species_name(species))
        
        return results
    
    def _extract_from_species_name(self, species_name: str) -> Dict[str, str]:
        """Fallback: extract genus from species name."""
        result = {
            'kingdom': 'Unknown',
            'phylum': 'Unknown',
            'class': 'Unknown',
            'order': 'Unknown',
            'family': 'Unknown',
            'genus': 'Unknown',
            'species': 'Unknown'
        }
        
        if species_name and species_name != '':
            parts = species_name.split()
            if parts:
                result['genus'] = parts[0]
                if len(parts) > 1:
                    result['species'] = ' '.join(parts[:2])
        
        return result


def save_no_hit_sequences(
    df_features: pl.DataFrame,
    df_blast: pl.DataFrame,
    analyzer: UnitigAnalyzer,
    output_dir: Path
):
    """Save sequences with no BLAST hits to FASTA file."""
    logger.info("Identifying sequences with no BLAST hits...")
    
    blast_dir = output_dir.parent.parent / 'blast_results'
    
    # Find features with no BLAST hits
    if len(df_blast) > 0:
        features_with_hits = set(df_blast['query_id'].to_list())
    else:
        features_with_hits = set()
    
    all_features = set(df_features['id'].to_list())
    no_hit_ids = all_features - features_with_hits
    
    logger.info(f"Found {len(no_hit_ids)} sequences with no BLAST hits out of {len(all_features)} total")
    
    if no_hit_ids:
        # Get feature indices for these IDs
        no_hit_features = df_features.filter(pl.col('id').is_in(list(no_hit_ids)))
        feature_indices = no_hit_features['feature_index'].unique().to_list()
        unitig_ids = no_hit_features['id'].unique().to_list()
        
        # Get sequences
        sequences_dict = analyzer.get_sequences_by_indices(feature_indices)
        
        # Write to FASTA
        no_hit_fasta = blast_dir / 'no_blast_hits.fasta'
        with open(no_hit_fasta, 'w') as f:
            for feat_idx, seq in sequences_dict.items():
                # Find unitig ID for this feature
                matching_rows = no_hit_features.filter(pl.col('feature_index') == feat_idx)
                if len(matching_rows) > 0:
                    unitig_id = matching_rows['id'][0]
                    f.write(f">{unitig_id}\n{seq}\n")
        
        logger.info(f"Saved {len(sequences_dict)} no-hit sequences to {no_hit_fasta}")
        return len(no_hit_ids), len(all_features)
    else:
        logger.info("All sequences had BLAST hits")
        return 0, len(all_features)


def create_blast_hit_rate_plot(
    num_no_hits: int,
    num_total: int,
    output_dir: Path
):
    """Create visualization of BLAST hit rate."""
    logger.info("Creating BLAST hit rate visualization...")
    
    num_hits = num_total - num_no_hits
    pct_hits = (num_hits / num_total * 100) if num_total > 0 else 0
    pct_no_hits = (num_no_hits / num_total * 100) if num_total > 0 else 0
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=['BLAST Hits Found', 'No BLAST Hits'],
        values=[num_hits, num_no_hits],
        hole=0.3,
        marker=dict(
            colors=['#2ecc71', '#e74c3c'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent+value',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>' +
                     'Count: %{value}<br>' +
                     'Percentage: %{percent}<br>' +
                     '<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text=f'BLAST Hit Rate for Top Features<br><sub>{num_hits}/{num_total} sequences with hits ({pct_hits:.1f}%)</sub>',
            x=0.5,
            xanchor='center'
        ),
        height=500,
        width=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        )
    )
    
    html_path = output_dir / 'blast_hit_rate.html'
    png_path = output_dir / 'blast_hit_rate.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=600, height=500, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")


def create_enhanced_summary_tables(
    df_features: pl.DataFrame,
    df_blast: pl.DataFrame,
    task_names: List[str],
    output_dir: Path,
    top_k: int = 20
):
    """Create enhanced summary tables with BLAST annotations."""
    logger.info("Creating enhanced summary tables with taxonomy...")
    
    tables_dir = output_dir.parent.parent / 'tables' / 'feature_analyses'
    
    # Initialize taxonomic resolver
    tax_resolver = TaxonomicResolver()
    
    # Add full taxonomy to BLAST results
    if len(df_blast) > 0 and 'best_taxid' in df_blast.columns:
        logger.info("Resolving taxonomic lineages (batched for performance)...")
        
        # Prepare batch input: list of (taxid, species) tuples
        taxid_species_pairs = [
            (row.get('best_taxid'), row.get('best_hit_species', ''))
            for row in df_blast.iter_rows(named=True)
        ]
        
        # Batch resolve all lineages at once
        lineages = tax_resolver.get_lineages_batch(taxid_species_pairs)
        
        # Build taxonomy data from batched results
        taxonomy_data = [
            {
                'query_id': row['query_id'],
                'kingdom': lineage['kingdom'],
                'phylum': lineage['phylum'],
                'class': lineage['class'],
                'order': lineage['order'],
                'family': lineage['family'],
                'genus': lineage['genus']
            }
            for row, lineage in zip(df_blast.iter_rows(named=True), lineages)
        ]
        
        df_taxonomy = pl.DataFrame(taxonomy_data)
        df_blast = df_blast.join(df_taxonomy, on='query_id', how='left')
    
    # Merge features with BLAST results
    df_merged = df_features.join(
        df_blast,
        left_on='id',
        right_on='query_id',
        how='left'
    )
    
    for task in task_names:
        task_data = df_merged.filter(pl.col('task') == task).sort('rank').head(top_k)
        
        # Select columns for output (include taxonomy if available)
        base_columns = [
            'rank',
            'feature_index',
            'id',
            'length',
            'gc_content',
            'complexity',
            'importance_score',
            'best_hit_species',
            'best_pident',
            'best_evalue',
            'best_bitscore'
        ]
        
        # Add taxonomy columns if available
        if 'genus' in task_data.columns:
            base_columns.extend(['phylum', 'family', 'genus'])
        
        # Select only columns that exist
        available_columns = [col for col in base_columns if col in task_data.columns]
        summary = task_data.select(available_columns)
        
        # Save as CSV
        csv_path = tables_dir / f'top_features_{task}_annotated.csv'
        summary.write_csv(csv_path)
        logger.info(f"Saved {csv_path}")
        
        # Save as markdown with better formatting
        md_path = tables_dir / f'top_features_{task}_annotated.md'
        with open(md_path, 'w') as f:
            f.write(f"# Top {top_k} Features for {task.replace('_', ' ').title()} (with BLAST Annotations)\n\n")
            
            # Build header based on available columns
            if 'genus' in summary.columns:
                f.write("| Rank | Feature | Unitig ID | Length | GC% | Complexity | Importance | Phylum | Family | Genus | Species | Identity% | E-value | Bitscore |\n")
                f.write("|------|---------|-----------|--------|-----|------------|------------|--------|--------|-------|---------|-----------|---------|----------|\n")
            else:
                f.write("| Rank | Feature | Unitig ID | Length | GC% | Complexity | Importance | Best BLAST Hit | Identity% | E-value | Bitscore |\n")
                f.write("|------|---------|-----------|--------|-----|------------|------------|----------------|-----------|---------|----------|\n")
            
            for row in summary.iter_rows(named=True):
                rank = row['rank']
                feat_idx = row['feature_index']
                uid = row['id']
                length = row['length']
                gc = row['gc_content']
                comp = row['complexity']
                imp = row['importance_score']
                
                species = row.get('best_hit_species')
                pident = row.get('best_pident')
                evalue = row.get('best_evalue')
                bits = row.get('best_bitscore')
                
                # Handle missing BLAST results
                species_str = species if species else "No hit"
                pident_str = f"{pident:.1f}" if pident else "N/A"
                evalue_str = f"{evalue:.2e}" if evalue else "N/A"
                bits_str = f"{bits:.1f}" if bits else "N/A"
                
                if 'genus' in row:
                    phylum = row.get('phylum', 'Unknown')
                    family = row.get('family', 'Unknown')
                    genus = row.get('genus', 'Unknown')
                    f.write(f"| {rank} | {feat_idx} | {uid} | {length} | {gc:.1f} | {comp:.3f} | {imp:.6f} | {phylum} | {family} | {genus} | {species_str} | {pident_str} | {evalue_str} | {bits_str} |\n")
                else:
                    f.write(f"| {rank} | {feat_idx} | {uid} | {length} | {gc:.1f} | {comp:.3f} | {imp:.6f} | {species_str} | {pident_str} | {evalue_str} | {bits_str} |\n")
        
        logger.info(f"Saved {md_path}")


def create_taxonomic_visualizations(
    df_features: pl.DataFrame,
    df_blast: pl.DataFrame,
    task_names: List[str],
    output_dir: Path,
    top_k: int = 20
):
    """Create taxonomic composition visualizations."""
    logger.info("Creating taxonomic visualizations...")
    
    # Add taxonomy to BLAST results if not already present
    if 'genus' not in df_blast.columns and len(df_blast) > 0:
        tax_resolver = TaxonomicResolver()
        
        # Prepare batch input
        taxid_species_pairs = [
            (row.get('best_taxid'), row.get('best_hit_species', ''))
            for row in df_blast.iter_rows(named=True)
        ]
        
        # Batch resolve all lineages
        lineages = tax_resolver.get_lineages_batch(taxid_species_pairs)
        
        # Add taxonomy data
        taxonomy_data = [
            {
                'query_id': row['query_id'],
                'kingdom': lineage['kingdom'],
                'phylum': lineage['phylum'],
                'class': lineage['class'],
                'order': lineage['order'],
                'family': lineage['family'],
                'genus': lineage['genus']
            }
            for row, lineage in zip(df_blast.iter_rows(named=True), lineages)
        ]
        
        df_taxonomy = pl.DataFrame(taxonomy_data)
        df_blast = df_blast.join(df_taxonomy, on='query_id', how='left')
    
    # Merge features with BLAST results
    df_merged = df_features.join(
        df_blast,
        left_on='id',
        right_on='query_id',
        how='left'
    )
    
    # 1. Taxonomic composition bar chart per task
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[task.replace('_', ' ').title() for task in task_names],
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    for idx, task in enumerate(task_names):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        task_data = df_merged.filter(pl.col('task') == task).head(top_k)
        
        # Use the genus column directly (already populated from taxonomy lookup)
        genera = []
        for genus in task_data['genus'].to_list():
            if genus and genus != '' and genus != 'Unknown':
                genera.append(genus)
            else:
                genera.append('No BLAST hit')
        
        # Count genus occurrences
        genus_counts = Counter(genera)
        
        # Sort by count
        sorted_genera = sorted(genus_counts.items(), key=lambda x: x[1], reverse=True)
        genera_names = [g[0] for g in sorted_genera]
        counts = [g[1] for g in sorted_genera]
        
        fig.add_trace(
            go.Bar(
                x=counts,
                y=genera_names,
                orientation='h',
                marker=dict(
                    color=counts,
                    colorscale='Viridis',
                    showscale=(idx == 0)
                ),
                name=task,
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text='Number of Features', row=row, col=col)
        fig.update_yaxes(title_text='Genus', row=row, col=col, tickfont=dict(size=9))
    
    fig.update_layout(
        title_text=f'Taxonomic Composition of Top {top_k} Features by Task',
        height=800,
        width=1400,
        template='plotly_white'
    )
    
    html_path = output_dir / 'taxonomic_composition_by_task.html'
    png_path = output_dir / 'taxonomic_composition_by_task.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=1400, height=800, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")
    
    # 2. Sunburst plot showing hierarchical taxonomy for each task
    for task in task_names:
        task_data = df_merged.filter(pl.col('task') == task).head(top_k)
        
        # Build hierarchical taxonomy: Phylum -> Family -> Genus
        # Track importance at each level
        phylum_imp = defaultdict(float)
        family_imp = defaultdict(lambda: defaultdict(float))  # phylum -> family -> imp
        genus_imp = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # phylum -> family -> genus -> imp
        
        for row in task_data.iter_rows(named=True):
            importance = row['importance_score']
            phylum = row.get('phylum') or 'Unknown Phylum'
            family = row.get('family') or 'Unknown Family'
            genus = row.get('genus') or 'Unknown Genus'
            
            phylum_imp[phylum] += importance
            family_imp[phylum][family] += importance
            genus_imp[phylum][family][genus] += importance
        
        # Prepare data for sunburst (hierarchical structure)
        # Use unique labels to avoid collisions
        labels = []
        parents = []
        values = []
        
        # Root
        root = task.replace('_', ' ').title()
        labels.append(root)
        parents.append('')
        values.append(0)  # Root gets 0, children sum up
        
        # Add phyla
        for phylum, phylum_total in phylum_imp.items():
            phylum_label = f"{phylum}"
            labels.append(phylum_label)
            parents.append(root)
            values.append(phylum_total)
            
            # Add families within this phylum
            for family, family_total in family_imp[phylum].items():
                # Make family labels unique by including phylum
                family_label = f"{family} ({phylum[:10]})"
                labels.append(family_label)
                parents.append(phylum_label)
                values.append(family_total)
                
                # Add genera within this family
                for genus, genus_total in genus_imp[phylum][family].items():
                    # Make genus labels unique by including family
                    genus_label = f"{genus}"
                    # If genus appears in multiple families, make it unique
                    if labels.count(genus_label) > 0:
                        genus_label = f"{genus} ({family[:10]})"
                    labels.append(genus_label)
                    parents.append(family_label)
                    values.append(genus_total)
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues='total',
            marker=dict(
                colorscale='Viridis',
                line=dict(width=2)
            ),
            hovertemplate='<b>%{label}</b><br>Total Importance: %{value:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Taxonomic Hierarchy: {task.replace("_", " ").title()}<br><sub>Phylum → Family → Genus</sub>',
            height=700,
            width=700,
            template='plotly_white'
        )
        
        html_path = output_dir / f'taxonomy_sunburst_{task}.html'
        png_path = output_dir / f'taxonomy_sunburst_{task}.png'
        fig.write_html(html_path)
        fig.write_image(png_path, width=700, height=700, scale=2)
        logger.info(f"Saved {html_path} and {png_path}")
    
    # 3. Heatmap: Genus × Task importance matrix
    logger.info("Creating genus-task importance heatmap...")
    
    # Collect all genera and their importance per task
    genus_task_importance = defaultdict(lambda: defaultdict(float))
    
    for task in task_names:
        task_data = df_merged.filter(pl.col('task') == task).head(top_k)
        
        for row in task_data.iter_rows(named=True):
            species = row['best_hit_species']
            importance = row['importance_score']
            
            if species and species != '':
                genus = species.split()[0]
            else:
                genus = 'Unknown'
            
            genus_task_importance[genus][task] += importance
    
    # Get top N genera by total importance
    top_n_genera = 15
    genus_totals = {g: sum(tasks.values()) for g, tasks in genus_task_importance.items()}
    top_genera = sorted(genus_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_genera]
    top_genera_names = [g[0] for g in top_genera]
    
    # Build matrix
    matrix = []
    for genus in top_genera_names:
        row = [genus_task_importance[genus][task] for task in task_names]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[t.replace('_', ' ').title() for t in task_names],
        y=top_genera_names,
        colorscale='Viridis',
        colorbar=dict(title='Total<br>Importance'),
        hovetemplate='<b>%{y}</b><br>%{x}<br>Importance: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {top_n_genera} Genera: Importance Across Tasks',
        xaxis_title='Classification Task',
        yaxis_title='Genus',
        height=600,
        width=800,
        template='plotly_white'
    )
    
    html_path = output_dir / 'genus_task_importance_heatmap.html'
    png_path = output_dir / 'genus_task_importance_heatmap.png'
    fig.write_html(html_path)
    fig.write_image(png_path, width=800, height=600, scale=2)
    logger.info(f"Saved {html_path} and {png_path}")


def main():
    parser = argparse.ArgumentParser(description='BLAST-based taxonomic annotation of important features')
    parser.add_argument('--config', type=str, default='configs/feature_analysis.yaml',
                       help='Path to feature analysis config file')
    parser.add_argument('--blast-db', type=str, default=None,
                       help='Path to BLAST database (overrides config file)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Number of top features to annotate per task (overrides config file)')
    parser.add_argument('--num-threads', type=int, default=None,
                       help='Number of threads for BLAST (overrides config file)')
    parser.add_argument('--skip-blast', action='store_true',
                       help='Skip BLAST run and use existing results')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n=== Starting feature annotation ===", flush=True)
    print(f"Loading configuration from {args.config}...", flush=True)
    logger.info(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    print(f"✓ Configuration loaded", flush=True)
    
    # Use config values if command-line args not provided
    if args.top_k is None:
        args.top_k = config.get('feature_importance', {}).get('top_k', 20)
    if args.num_threads is None:
        args.num_threads = config.get('blast', {}).get('num_threads', 4)
    
    print(f"  Using top_k={args.top_k} features per task", flush=True)
    print(f"  Using num_threads={args.num_threads} for BLAST", flush=True)
    
    # Setup paths
    print(f"Setting up paths...", flush=True)
    output_dir = Path(config['output']['figures_dir'])
    tables_dir = Path(config['output']['tables_dir'])
    print(f"  Tables directory: {tables_dir}", flush=True)
    print(f"  Output directory: {output_dir}", flush=True)
    
    # Get unitigs.fa path
    matrix_path = Path(config['data']['matrix_path'])
    unitigs_fa = matrix_path.parent / 'unitigs.fa'
    print(f"  Unitigs file: {unitigs_fa}", flush=True)
    
    if not unitigs_fa.exists():
        raise FileNotFoundError(f"unitigs.fa not found at {unitigs_fa}")
    print(f"✓ All paths valid", flush=True)
    
    # Load feature tables with sequences (using all available tasks)
    print(f"\nLooking for feature tables in {tables_dir}...", flush=True)
    logger.info("Loading feature tables...")
    # Get all feature table files to infer task names
    print(f"Searching for top_features_*_with_sequences.csv files...", flush=True)
    feature_files = list(tables_dir.glob('top_features_*_with_sequences.csv'))
    print(f"Found {len(feature_files)} feature files", flush=True)
    if not feature_files:
        raise FileNotFoundError(f"No feature tables found in {tables_dir}")
    
    # Infer task names from file names
    task_names = []
    for f in feature_files:
        # Extract task name from filename: top_features_<task>_with_sequences.csv
        task = f.stem.replace('top_features_', '').replace('_with_sequences', '')
        task_names.append(task)
    
    print(f"Tasks found: {task_names}", flush=True)
    logger.info(f"Found tasks: {task_names}")
    
    print(f"\nLoading feature tables for {len(task_names)} tasks (top_k={args.top_k})...", flush=True)
    logger.info(f"Loading feature tables for {len(task_names)} tasks with top_k={args.top_k}...")
    df_features = load_feature_tables_with_sequences(tables_dir, task_names, args.top_k)
    print(f"✓ Loaded {len(df_features)} total feature records", flush=True)
    logger.info(f"Successfully loaded {len(df_features)} total feature records")
    
    # Initialize UnitigAnalyzer
    print(f"\nInitializing UnitigAnalyzer with {unitigs_fa}...", flush=True)
    logger.info("Initializing UnitigAnalyzer...")
    logger.info(f"Unitigs file: {unitigs_fa}")
    analyzer = UnitigAnalyzer(str(unitigs_fa))
    print(f"✓ UnitigAnalyzer initialized", flush=True)
    logger.info("UnitigAnalyzer initialized successfully")
    
    # Get BLAST database path (command line overrides config)
    blast_db = args.blast_db
    if blast_db is None:
        # Try to get from config
        if 'blast' in config and 'database_path' in config['blast']:
            blast_db = config['blast']['database_path']
            logger.info(f"Using BLAST database from config: {blast_db}")
        else:
            raise ValueError("BLAST database path must be specified via --blast-db or in config file under blast.database_path")
    else:
        logger.info(f"Using BLAST database from command line: {blast_db}")
    
    # Suggest RefSeq bacteria if using nt (much faster, more relevant for microbiome)
    if 'nt' in blast_db and '/refseq' not in blast_db:
        logger.warning("⚠️  You're using the 'nt' database which is very large and slow.")
        logger.warning("⚠️  Consider using RefSeq Bacteria instead for 10-100x speedup:")
        logger.warning("⚠️  Example: /local/databases/rel/refseq/bacteria/current/blast+/2.10.0/refseq_bacteria")
        logger.warning("⚠️  Or update your config file: blast.database_path")
    
    # BLAST analysis
    print(f"\n=== BLAST Analysis ===", flush=True)
    blast_dir = output_dir.parent.parent / 'blast_results'
    blast_dir.mkdir(exist_ok=True, parents=True)
    
    query_fasta = blast_dir / 'top_features.fasta'
    blast_output = blast_dir / 'blast_results.txt'
    
    # Check if we should run BLAST
    run_blast_now = not args.skip_blast
    
    if run_blast_now and blast_output.exists() and blast_output.stat().st_size > 0:
        # Found existing results
        print(f"\n⚠️  Found existing BLAST results: {blast_output}", flush=True)
        print(f"    File size: {blast_output.stat().st_size:,} bytes", flush=True)
        print(f"    To force rerun, delete this file first:", flush=True)
        print(f"    rm {blast_output}", flush=True)
        print(f"\nUsing existing BLAST results...", flush=True)
        logger.info(f"Using existing BLAST results from {blast_output}")
        run_blast_now = False
    
    if run_blast_now:
        # Extract sequences and run BLAST
        print(f"\nExtracting {len(df_features)} sequences for BLAST...", flush=True)
        logger.info(f"Extracting {len(df_features)} sequences for BLAST...")
        id_to_index = extract_sequences_for_blast(df_features, analyzer, query_fasta)
        print(f"✓ Sequences written to: {query_fasta}", flush=True)
        logger.info(f"Sequences written to: {query_fasta}")
        
        print(f"\nRunning BLAST against {blast_db}...", flush=True)
        print(f"This may take 10-30 minutes with the nt database...", flush=True)
        success = run_blast(
            query_fasta=query_fasta,
            blast_db=blast_db,
            output_file=blast_output,
            num_threads=args.num_threads
        )
        
        if not success:
            logger.error("BLAST failed. Exiting.")
            return
    else:
        if args.skip_blast:
            print(f"\nSkipping BLAST (--skip-blast flag set)", flush=True)
            logger.info("Skipping BLAST run (--skip-blast flag), using existing results...")
    
    # Parse BLAST results
    print(f"\n=== Parsing BLAST Results ===", flush=True)
    print(f"Reading from: {blast_output}", flush=True)
    df_blast = parse_blast_results(blast_output)
    print(f"✓ Parsed {len(df_blast)} BLAST hits", flush=True)
    
    if len(df_blast) == 0:
        logger.warning("No BLAST results found. Check if BLAST ran successfully.")
        logger.warning("Continuing with empty annotations...")
        df_blast = pl.DataFrame({
            'query_id': [],
            'best_hit_species': [],
            'best_hit_title': [],
            'best_pident': [],
            'best_length': [],
            'best_evalue': [],
            'best_bitscore': [],
            'best_taxid': [],
            'num_hits': []
        })
    
    # Save sequences with no BLAST hits
    print(f"\n=== Generating Outputs ===", flush=True)
    print(f"Saving sequences with no BLAST hits...", flush=True)
    num_no_hits, num_total = save_no_hit_sequences(df_features, df_blast, analyzer, output_dir)
    
    # Create BLAST hit rate visualization
    print(f"Creating BLAST hit rate visualization...", flush=True)
    create_blast_hit_rate_plot(num_no_hits, num_total, output_dir)
    
    # Create enhanced tables
    print(f"Creating enhanced annotation tables...", flush=True)
    create_enhanced_summary_tables(df_features, df_blast, task_names, output_dir, args.top_k)
    
    # Create taxonomic visualizations
    if len(df_blast) > 0:
        print(f"Creating taxonomic visualizations...", flush=True)
        create_taxonomic_visualizations(df_features, df_blast, task_names, output_dir, args.top_k)
    else:
        logger.warning("Skipping taxonomic visualizations (no BLAST results)")
    
    print(f"\n{'='*60}", flush=True)
    print(f"✅ Feature annotation complete!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"BLAST results:     {blast_output}", flush=True)
    print(f"No-hit sequences:  {blast_dir / 'no_blast_hits.fasta'}", flush=True)
    print(f"Enhanced tables:   {tables_dir}", flush=True)
    print(f"Taxonomic figures: {output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    logger.info("\n✅ Feature annotation complete!")
    logger.info(f"BLAST results: {blast_output}")
    logger.info(f"No-hit sequences: {blast_dir / 'no_blast_hits.fasta'}")
    logger.info(f"Enhanced tables: {tables_dir}")
    logger.info(f"Taxonomic figures: {output_dir}")


if __name__ == "__main__":
    main()
