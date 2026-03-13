#!/usr/bin/env python3
"""
BLAST All Unitig Features - Complete Feature Annotation for Reviewers
======================================================================

Annotates ALL 107,480 unitig features (not just top important ones) to provide
comprehensive statistics about the biological nature of features:
- % of features with BLAST hits
- Taxonomic breakdown across all features
- Summary of kingdoms/phyla/genera represented

This addresses potential reviewer questions:
- "What are these features biologically?"
- "How many can be taxonomically assigned?"
- "What organisms do they come from?"

REQUIRES:
---------
- BLAST+ (module load blast)
- BLAST database (e.g., /local/databases/index/blast+/nt)
- unitigs.fa from matrix generation

USAGE:
------
# For full run (SLURM recommended - takes ~2-24 hours on nt database)
sbatch scripts/feature_analysis/run_blast_all_features.sbatch

# For testing (first 1000 features)
python scripts/feature_analysis/04_blast_all_features.py \\
    --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \\
    --blast-db /local/databases/index/blast+/nt \\
    --output results/feature_analysis/all_features_blast \\
    --num-threads 16 \\
    --max-features 1000

# For full run (all features, with checkpointing)
python scripts/feature_analysis/04_blast_all_features.py \\
    --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \\
    --blast-db /local/databases/index/blast+/nt \\
    --output results/feature_analysis/all_features_blast \\
    --num-threads 32 \\
    --chunk-size 10000  # Process 10k features at a time

# Resume from failure (automatically skips completed chunks)
python scripts/feature_analysis/04_blast_all_features.py \\
    --unitigs-fa data/matrices/large_matrix_3070_with_frac/unitigs.fa \\
    --blast-db /local/databases/index/blast+/nt \\
    --output results/feature_analysis/all_features_blast \\
    --num-threads 32 \\
    --resume

OUTPUT:
-------
- chunks/chunk_*.fa: Query sequences split into chunks (cached)
- chunks/chunk_*_blast.txt: BLAST results per chunk (cached)
- blast_results.txt: Combined BLAST output from all chunks
- blast_summary.json: Summary statistics
- taxonomic_breakdown.csv: Counts by taxonomic rank
- hit_statistics.txt: Human-readable summary
- progress.json: Checkpointing info for resume
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def count_sequences_in_fasta(fasta_path: Path) -> int:
    """Count number of sequences in FASTA file."""
    count = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith('>'):
                count += 1
    return count


def split_fasta_into_chunks(
    input_fasta: Path,
    output_dir: Path,
    chunk_size: int,
    max_features: Optional[int] = None
) -> List[Path]:
    """
    Split FASTA into chunks for parallel/resumable processing.
    
    Returns list of chunk file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunk_files = []
    current_chunk = 0
    sequences_in_chunk = 0
    total_sequences = 0
    
    chunk_file = output_dir / f"chunk_{current_chunk:04d}.fa"
    chunk_files.append(chunk_file)
    outfile = open(chunk_file, 'w')
    
    logger.info(f"Splitting {input_fasta} into chunks of {chunk_size} sequences")
    
    with open(input_fasta) as infile:
        for line in infile:
            if line.startswith('>'):
                # Check if we've hit the limit
                if max_features and total_sequences >= max_features:
                    break
                
                # Start new chunk if current one is full
                if sequences_in_chunk >= chunk_size:
                    outfile.close()
                    current_chunk += 1
                    sequences_in_chunk = 0
                    chunk_file = output_dir / f"chunk_{current_chunk:04d}.fa"
                    chunk_files.append(chunk_file)
                    outfile = open(chunk_file, 'w')
                    logger.info(f"  Created chunk {current_chunk}: {chunk_file.name}")
                
                sequences_in_chunk += 1
                total_sequences += 1
            
            outfile.write(line)
    
    outfile.close()
    
    logger.info(f"Split {total_sequences:,} sequences into {len(chunk_files)} chunks")
    return chunk_files


def load_progress(output_dir: Path) -> Dict:
    """Load progress from previous run."""
    progress_file = output_dir / 'progress.json'
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {'completed_chunks': [], 'failed_chunks': []}


def save_progress(output_dir: Path, progress: Dict):
    """Save progress for resume capability."""
    progress_file = output_dir / 'progress.json'
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def extract_subset_fasta(input_fasta: Path, output_fasta: Path, max_features: int):
    """Extract first N sequences from FASTA for testing."""
    logger.info(f"Extracting first {max_features} sequences from {input_fasta}")
    
    count = 0
    with open(input_fasta) as infile, open(output_fasta, 'w') as outfile:
        write_seq = False
        for line in infile:
            if line.startswith('>'):
                if count >= max_features:
                    break
                count += 1
                write_seq = True
            if write_seq:
                outfile.write(line)
    
    logger.info(f"Extracted {count} sequences to {output_fasta}")
    return count


def run_blast(
    query_fasta: Path,
    blast_db: str,
    output_file: Path,
    num_threads: int = 32,
    evalue: float = 1e-5,
    chunk_name: str = ""
) -> bool:
    """
    Run BLAST analysis on query sequences.
    
    Uses optimized parameters for large-scale annotation:
    - max_target_seqs=1: Only keep best hit (faster, smaller output)
    - task=megablast: Faster for highly similar sequences
    """
    chunk_label = f" [{chunk_name}]" if chunk_name else ""
    logger.info(f"Running BLAST{chunk_label} against {blast_db}...")
    logger.info(f"  Query: {query_fasta}")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Threads: {num_threads}")
    
    cmd = [
        'blastn',
        '-query', str(query_fasta),
        '-db', blast_db,
        '-out', str(output_file),
        '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids stitle',
        '-num_threads', str(num_threads),
        '-max_target_seqs', '1',  # Only best hit per query
        '-evalue', str(evalue),
        '-task', 'megablast'  # Faster for similar sequences
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=86400  # 24 hour timeout per chunk
        )
        logger.info(f"  ✓ BLAST{chunk_label} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ BLAST{chunk_label} timed out after 24 hours")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"  ✗ BLAST{chunk_label} failed: {e}")
        return False


def combine_blast_results(chunk_files: List[Path], output_file: Path):
    """Combine BLAST results from multiple chunks into single file."""
    logger.info(f"Combining {len(chunk_files)} BLAST result files...")
    
    total_lines = 0
    with open(output_file, 'w') as outfile:
        for chunk_file in sorted(chunk_files):
            if not chunk_file.exists():
                logger.warning(f"  Skipping missing file: {chunk_file}")
                continue
            
            chunk_lines = 0
            with open(chunk_file) as infile:
                for line in infile:
                    outfile.write(line)
                    chunk_lines += 1
            
            total_lines += chunk_lines
            logger.info(f"  Added {chunk_lines:,} hits from {chunk_file.name}")
    
    logger.info(f"Combined {total_lines:,} total BLAST hits to {output_file}")
    return total_lines


KINGDOM_KEYWORDS = {
    'Bacteria': ['bacterium', 'bacteria', 'bacillus', 'streptococcus', 'staphylococcus',
                 'escherichia', 'pseudomonas', 'clostridium', 'mycobacterium'],
    'Archaea':  ['archaeon', 'archaea', 'methanobrevibacter', 'haloferax'],
    'Eukaryota': ['human', 'homo sapiens', 'mouse', 'rat', 'plant', 'arabidopsis',
                  'saccharomyces', 'drosophila', 'caenorhabditis'],
    'Viruses':  ['virus', 'phage', 'bacteriophage'],
}


def _assign_kingdom(description: str) -> str:
    """Assign kingdom from BLAST hit description via keyword matching."""
    d = description.lower()
    for kingdom, keywords in KINGDOM_KEYWORDS.items():
        if any(kw in d for kw in keywords):
            return kingdom
    return 'Unknown'


def parse_blast_results(blast_file: Path) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Parse BLAST results and generate summary statistics.

    For each query feature:
    - ``best_hits``: keeps the single hit with the highest bitscore (for pident stats).
    - ``best_kingdom_hits``: keeps the highest-bitscore hit whose kingdom is not Unknown
      (for taxonomy stats).  Falls back to Unknown if no annotated hit exists.

    Returns:
        - List of best-hit dicts (one per unique query, for pident stats)
        - List of best-annotated-hit dicts (one per query with a known kingdom)
        - Summary statistics dict
    """
    logger.info(f"Parsing BLAST results from {blast_file}")

    # BLAST output columns:
    # qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore staxids stitle

    best_hits: Dict[str, Dict] = {}         # query_id -> best hit by bitscore (overall)
    best_kingdom_hits: Dict[str, Dict] = {} # query_id -> best hit with non-Unknown kingdom

    with open(blast_file) as f:
        for line in f:
            if not line.strip():
                continue

            fields = line.strip().split('\t')
            if len(fields) < 14:
                logger.warning(f"Skipping malformed line: {line[:100]}")
                continue

            query_id = fields[0]
            bitscore  = float(fields[11])
            description = fields[13]
            kingdom = _assign_kingdom(description)

            hit = {
                'query_id':   query_id,
                'subject_id': fields[1],
                'pident':     float(fields[2]),
                'length':     int(fields[3]),
                'evalue':     float(fields[10]),
                'bitscore':   bitscore,
                'taxids':     fields[12] if fields[12] != 'N/A' else None,
                'description': description,
                'kingdom':    kingdom,
            }

            # Best hit overall (for pident)
            if query_id not in best_hits or bitscore > best_hits[query_id]['bitscore']:
                best_hits[query_id] = hit

            # Best hit with a known kingdom (for taxonomy)
            if kingdom != 'Unknown':
                if query_id not in best_kingdom_hits or bitscore > best_kingdom_hits[query_id]['bitscore']:
                    best_kingdom_hits[query_id] = hit

    hits = list(best_hits.values())
    kingdom_hits = list(best_kingdom_hits.values())
    n_known = len(kingdom_hits)
    n_unknown = len(hits) - n_known

    logger.info(f"Parsed {len(hits):,} best hits (one per unique query feature)")
    logger.info(f"  With known kingdom: {n_known:,} | Unknown only: {n_unknown:,}")

    # pident stats use the single best hit per feature
    stats = {
        'total_hits': len(hits),
        'unique_queries_with_hits': len(hits),
        'pident_ranges': {
            '>=95%': sum(1 for h in hits if h['pident'] >= 95),
            '90-95%': sum(1 for h in hits if 90 <= h['pident'] < 95),
            '80-90%': sum(1 for h in hits if 80 <= h['pident'] < 90),
            '<80%':   sum(1 for h in hits if h['pident'] < 80),
        },
        'evalue_ranges': {
            '<=1e-50':         sum(1 for h in hits if h['evalue'] <= 1e-50),
            '1e-50 to 1e-10':  sum(1 for h in hits if 1e-50 < h['evalue'] <= 1e-10),
            '1e-10 to 1e-5':   sum(1 for h in hits if 1e-10 < h['evalue'] <= 1e-5),
        },
        'note': 'One best hit per feature (highest bitscore); pident from best overall hit',
    }

    return hits, kingdom_hits, stats


def extract_taxonomy_from_hits(
    hits: List[Dict],
    kingdom_hits: Optional[List[Dict]] = None,
    n_unknown: int = 0,
) -> Dict:
    """
    Extract taxonomic information from BLAST hit descriptions.

    Parameters
    ----------
    hits : list
        All best-per-feature hits (used for genus extraction).
    kingdom_hits : list, optional
        Best non-Unknown-kingdom hit per feature.  When provided, kingdom
        counts are taken from these; otherwise falls back to keyword-matching
        all ``hits``.
    n_unknown : int
        Number of features whose best hit yielded no known kingdom (counted
        in kingdom_counts['Unknown']).
    """
    logger.info("Extracting taxonomy from hit descriptions...")

    # Use pre-filtered kingdom hits when available
    kingdom_source = kingdom_hits if kingdom_hits is not None else hits

    genera = []
    for hit in kingdom_source:
        desc = hit['description']
        words = desc.split()
        if words:
            potential_genus = words[0].capitalize()
            if len(potential_genus) > 2:
                genera.append(potential_genus)

    # Kingdom counts: from 'kingdom' field (set in parse_blast_results) or keyword-match
    if kingdom_hits is not None:
        kingdom_counts: Counter = Counter(h['kingdom'] for h in kingdom_source)
        kingdom_counts['Unknown'] = n_unknown
    else:
        kingdoms = []
        for hit in kingdom_source:
            kingdoms.append(_assign_kingdom(hit['description']))
        kingdom_counts = Counter(kingdoms)

    taxonomy_stats = {
        'top_genera': dict(Counter(genera).most_common(20)),
        'kingdom_counts': dict(kingdom_counts),
        'features_with_known_kingdom': len(kingdom_source) if kingdom_hits is not None else None,
        'unique_genera': len(set(genera)),
        'total_assigned': len(genera),
        'note': 'Kingdom uses best non-Unknown hit per feature; Unknown = no known-kingdom hit found',
    }

    return taxonomy_stats


def write_summary(
    output_dir: Path,
    total_features: int,
    stats: Dict,
    taxonomy: Dict,
    blast_file: Path
):
    """Write comprehensive summary files."""
    
    # JSON summary
    summary = {
        'total_features': total_features,
        'features_with_blast_hits': stats['unique_queries_with_hits'],
        'features_with_known_kingdom': taxonomy.get('features_with_known_kingdom'),
        'hit_rate_percent': round(100 * stats['unique_queries_with_hits'] / total_features, 2),
        'blast_statistics': stats,
        'taxonomy': taxonomy,
        'blast_results_file': str(blast_file)
    }
    
    summary_file = output_dir / 'blast_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary to {summary_file}")
    
    # Human-readable text summary
    text_file = output_dir / 'hit_statistics.txt'
    with open(text_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BLAST Annotation Summary - All Unitig Features\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total features queried: {total_features:,}\n")
        f.write(f"Features with BLAST hits: {stats['unique_queries_with_hits']:,}\n")
        f.write(f"Hit rate: {summary['hit_rate_percent']}%\n\n")
        
        f.write("Identity Distribution:\n")
        for range_name, count in stats['pident_ranges'].items():
            pct = 100 * count / stats['total_hits'] if stats['total_hits'] > 0 else 0
            f.write(f"  {range_name}: {count:,} ({pct:.1f}%)\n")
        
        f.write("\nE-value Distribution:\n")
        for range_name, count in stats['evalue_ranges'].items():
            pct = 100 * count / stats['total_hits'] if stats['total_hits'] > 0 else 0
            f.write(f"  {range_name}: {count:,} ({pct:.1f}%)\n")
        
        f.write("\nKingdom Distribution:\n")
        for kingdom, count in sorted(taxonomy['kingdom_counts'].items(), 
                                     key=lambda x: x[1], reverse=True):
            pct = 100 * count / stats['total_hits'] if stats['total_hits'] > 0 else 0
            f.write(f"  {kingdom}: {count:,} ({pct:.1f}%)\n")
        
        f.write(f"\nUnique genera identified: {taxonomy['unique_genera']:,}\n")
        f.write("\nTop 20 Genera:\n")
        for genus, count in list(taxonomy['top_genera'].items())[:20]:
            pct = 100 * count / stats['total_hits'] if stats['total_hits'] > 0 else 0
            f.write(f"  {genus}: {count:,} ({pct:.1f}%)\n")
    
    logger.info(f"Wrote human-readable summary to {text_file}")
    
    # CSV for taxonomy breakdown
    csv_file = output_dir / 'taxonomic_breakdown.csv'
    with open(csv_file, 'w') as f:
        f.write("Genus,Count,Percentage\n")
        for genus, count in taxonomy['top_genera'].items():
            pct = 100 * count / stats['total_hits'] if stats['total_hits'] > 0 else 0
            f.write(f"{genus},{count},{pct:.2f}\n")
    logger.info(f"Wrote taxonomic breakdown to {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='BLAST all unitig features for comprehensive annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--unitigs-fa', type=Path, required=True,
                       help='Path to unitigs.fa file')
    parser.add_argument('--blast-db', type=str, required=True,
                       help='Path to BLAST database (e.g., /local/databases/index/blast+/nt)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--num-threads', type=int, default=32,
                       help='Number of threads for BLAST (default: 32)')
    parser.add_argument('--evalue', type=float, default=1e-5,
                       help='E-value cutoff (default: 1e-5)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Features per chunk for checkpointing (default: 10000)')
    parser.add_argument('--max-features', type=int, default=None,
                       help='Maximum number of features to process (for testing)')
    parser.add_argument('--skip-blast', action='store_true',
                       help='Skip BLAST and only parse existing results')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous run (skip completed chunks)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output / 'chunks'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = args.output / 'blast_all_features.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("BLAST All Features - Comprehensive Unitig Annotation with Checkpointing")
    logger.info("=" * 70)
    
    # Verify input file
    if not args.unitigs_fa.exists():
        logger.error(f"Input file not found: {args.unitigs_fa}")
        return 1
    
    # Count total sequences
    total_features = count_sequences_in_fasta(args.unitigs_fa)
    logger.info(f"Total sequences in {args.unitigs_fa}: {total_features:,}")
    
    # Load progress if resuming
    progress = load_progress(args.output)
    if args.resume and progress['completed_chunks']:
        logger.info(f"Resuming: {len(progress['completed_chunks'])} chunks already completed")
    
    # Split into chunks or use existing chunks
    chunk_query_files = list(chunks_dir.glob('chunk_*.fa'))
    
    if not chunk_query_files or args.max_features:
        # Create new chunks
        chunk_query_files = split_fasta_into_chunks(
            input_fasta=args.unitigs_fa,
            output_dir=chunks_dir,
            chunk_size=args.chunk_size,
            max_features=args.max_features
        )
    else:
        logger.info(f"Using existing {len(chunk_query_files)} chunk files")
        chunk_query_files = sorted(chunk_query_files)
    
    features_to_blast = sum(count_sequences_in_fasta(f) for f in chunk_query_files)
    logger.info(f"Total features to BLAST: {features_to_blast:,} in {len(chunk_query_files)} chunks")
    
    # BLAST each chunk
    blast_result_files = []
    
    if not args.skip_blast:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Running BLAST on {len(chunk_query_files)} chunks")
        logger.info("=" * 70)
        
        for i, query_chunk in enumerate(chunk_query_files):
            chunk_name = query_chunk.stem
            blast_output = chunks_dir / f"{chunk_name}_blast.txt"
            
            # Skip if already completed and resuming
            if args.resume and chunk_name in progress['completed_chunks']:
                logger.info(f"[{i+1}/{len(chunk_query_files)}] Skipping {chunk_name} (already completed)")
                blast_result_files.append(blast_output)
                continue
            
            # Run BLAST on this chunk
            logger.info(f"[{i+1}/{len(chunk_query_files)}] Processing {chunk_name}...")
            success = run_blast(
                query_fasta=query_chunk,
                blast_db=args.blast_db,
                output_file=blast_output,
                num_threads=args.num_threads,
                evalue=args.evalue,
                chunk_name=chunk_name
            )
            
            if success:
                blast_result_files.append(blast_output)
                progress['completed_chunks'].append(chunk_name)
                save_progress(args.output, progress)
            else:
                logger.error(f"  Failed to BLAST {chunk_name}")
                progress['failed_chunks'].append(chunk_name)
                save_progress(args.output, progress)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"BLAST Complete: {len(progress['completed_chunks'])}/{len(chunk_query_files)} chunks succeeded")
        if progress['failed_chunks']:
            logger.warning(f"Failed chunks: {progress['failed_chunks']}")
        logger.info("=" * 70)
        
    else:
        # Skip BLAST, use existing chunk results
        logger.info("Skipping BLAST (using existing chunk results)")
        blast_result_files = sorted(chunks_dir.glob('chunk_*_blast.txt'))
        if not blast_result_files:
            logger.error(f"No chunk BLAST results found in {chunks_dir}")
            return 1
    
    # Combine chunk results
    combined_blast_output = args.output / 'blast_results.txt'
    combine_blast_results(blast_result_files, combined_blast_output)
    
    # Parse combined results
    hits, kingdom_hits, stats = parse_blast_results(combined_blast_output)

    if not hits:
        logger.warning("No BLAST hits found!")
        return 1

    # Extract taxonomy (use best non-Unknown kingdom hit per feature)
    n_unknown = len(hits) - len(kingdom_hits)
    taxonomy = extract_taxonomy_from_hits(hits, kingdom_hits=kingdom_hits, n_unknown=n_unknown)
    
    # Write summaries
    write_summary(
        output_dir=args.output,
        total_features=features_to_blast,
        stats=stats,
        taxonomy=taxonomy,
        blast_file=combined_blast_output
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total features: {features_to_blast:,}")
    logger.info(f"Features with hits: {stats['unique_queries_with_hits']:,}")
    hit_rate = 100 * stats['unique_queries_with_hits'] / features_to_blast
    logger.info(f"Hit rate: {hit_rate:.2f}%")
    logger.info(f"Unique genera: {taxonomy['unique_genera']:,}")
    logger.info("=" * 70)
    
    logger.info(f"\nAll results saved to: {args.output}")
    logger.info("Chunk files preserved in chunks/ for future resume")
    logger.info("Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
