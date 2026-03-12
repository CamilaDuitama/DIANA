#!/usr/bin/env python3
"""
Generate PCA analysis of unitig matrix with task label annotations.

This script:
1. Loads the unitig fraction matrix (3,070 samples × 107,480 unitigs)
2. Performs PCA dimensionality reduction
3. Creates visualizations with samples colored by task labels
4. Saves PCA model for future projections

Output:
- paper/figures/final/sup_03_pca_*.png/html - PCA plots by task
- paper/figures/final/sup_03_pca_scree_plot.png - Explained variance
- paper/figures/final/sup_04_pca_loadings_*.png/html - PCA loading plots
- models/pca_reference.pkl - Saved PCA model for projecting new samples
"""

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy import stats
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from pathlib import Path
import pickle
import sys
import logging
import json
import os
import subprocess
from tqdm import tqdm

# Import local config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'configs'))
from paper_config import PATHS, TASKS, PLOT_CONFIG

# Configure multithreading
N_JOBS = int(os.environ.get('OMP_NUM_THREADS', -1))  # Use all available cores by default

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(PATHS['figures_dir']).parent / 'pca_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_unitig_matrix(standardize=False):
    """Load the unitig fraction matrix with validation and optional standardization."""
    logger.info("Loading unitig matrix...")
    matrix_path = Path("data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat")
    
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    
    # Matrix format: first column is unitig ID, remaining columns are sample fractions
    # Shape: 107,480 unitigs × 3,071 columns (1 ID + 3,070 samples)
    logger.info(f"  Reading matrix from {matrix_path} (this may take a minute)...")
    try:
        data = np.loadtxt(matrix_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load matrix: {e}")
    
    # Split into unitig IDs and fraction matrix
    unitig_ids = data[:, 0].astype(int)  # First column = unitig IDs
    matrix = data[:, 1:]  # Remaining columns = fractions
    
    # CRITICAL VALIDATION: Ensure unitig IDs are excluded from PCA input
    logger.info("  Validating unitig ID extraction...")
    logger.info(f"    Unitig ID range: {unitig_ids.min():,} to {unitig_ids.max():,}")
    logger.info(f"    Matrix value range: {matrix.min():.4f} to {matrix.max():.4f}")
    
    # Unitig IDs should be large integers (100k+), matrix values should be fractions (0-1)
    if unitig_ids.min() < 100:  # IDs should be large
        raise ValueError(f"Unitig IDs seem wrong - min ID is {unitig_ids.min()}")
    if matrix.max() > 10:  # Fractions shouldn't exceed 1 by much
        logger.warning(f"  Matrix has values > 10 (max={matrix.max():.2f}) - are these fractions?")
    if np.any(unitig_ids == matrix[:, 0]):
        raise ValueError("CRITICAL ERROR: Unitig IDs still in matrix! They will affect PCA!")
    
    logger.info(f"    ✓ Confirmed: Unitig IDs excluded from PCA matrix")
    
    # Transpose to get samples × unitigs
    matrix = matrix.T
    
    logger.info(f"  Matrix shape: {matrix.shape} (samples × unitigs)")
    logger.info(f"  Unitigs: {len(unitig_ids):,}")
    
    # Validate expected dimensions
    if matrix.shape[0] != 3070:
        logger.warning(f"  Expected 3,070 samples but got {matrix.shape[0]}")
    if matrix.shape[1] != len(unitig_ids):
        raise ValueError(f"Column mismatch: {matrix.shape[1]} != {len(unitig_ids)}")
    
    # Get sample IDs from metadata (train + test) - PRESERVE ORDER
    # CRITICAL FIX: Use kmtricks.fof order, NOT metadata concatenation!
    logger.info("\n  CRITICAL: Loading sample order from kmtricks.fof...")
    fof_path = Path("data/matrices/large_matrix_3070_with_frac/kmer_matrix/kmtricks.fof")
    
    if not fof_path.exists():
        raise FileNotFoundError(
            f"kmtricks.fof not found at {fof_path}! Cannot determine sample order."
        )
    
    # Parse kmtricks.fof to get ACTUAL sample order used by matrix
    sample_ids = []
    with open(fof_path) as f:
        for line in f:
            # Format: "SAMPLE_ID : /path/to/file"
            if ':' in line:
                sample_id = line.split(':')[0].strip()
                sample_ids.append(sample_id)
    
    logger.info(f"    ✓ Loaded {len(sample_ids)} samples from kmtricks.fof (ACTUAL matrix order)")
    
    # Verify count matches
    if len(sample_ids) != matrix.shape[0]:
        raise ValueError(
            f"Sample count mismatch! "
            f"Matrix has {matrix.shape[0]} rows but kmtricks.fof has {len(sample_ids)} samples"
        )
    
    # Load metadata to verify coverage
    train_meta = pl.read_csv(PATHS['train_metadata'], separator="\t")
    test_meta = pl.read_csv(PATHS['test_metadata'], separator="\t")
    train_ids = set(train_meta['Run_accession'].to_list())
    test_ids = set(test_meta['Run_accession'].to_list())
    all_meta_ids = train_ids | test_ids
    
    # Check coverage
    missing_from_meta = [sid for sid in sample_ids if sid not in all_meta_ids]
    if missing_from_meta:
        logger.warning(f"    {len(missing_from_meta)} samples in fof but not in metadata: {missing_from_meta[:5]}...")
    
    # Analyze train/test distribution
    n_train_in_matrix = sum(1 for sid in sample_ids if sid in train_ids)
    n_test_in_matrix = sum(1 for sid in sample_ids if sid in test_ids)
    
    logger.info(f"    Sample distribution in matrix:")
    logger.info(f"      Train samples: {n_train_in_matrix} ({n_train_in_matrix/len(sample_ids)*100:.1f}%)")
    logger.info(f"      Test samples: {n_test_in_matrix} ({n_test_in_matrix/len(sample_ids)*100:.1f}%)")
    logger.info(f"    ✓ Sample order matches kmtricks.fof (matrix generation order)")
    
    # Remove the incorrect verification section entirely since we're now USING the fof order
    # Validate sample count matches
    if len(sample_ids) != matrix.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {len(sample_ids)} IDs but matrix has {matrix.shape[0]} rows"
        )
    
    logger.info(f"  Loaded {len(sample_ids)} sample IDs")
    
    # Data validation
    logger.info("Validating data quality...")
    nan_count = np.isnan(matrix).sum()
    inf_count = np.isinf(matrix).sum()
    if nan_count > 0:
        logger.warning(f"  Found {nan_count} NaN values - replacing with 0")
        matrix = np.nan_to_num(matrix, nan=0.0)
    if inf_count > 0:
        logger.warning(f"  Found {inf_count} infinite values - replacing with 0")
        matrix = np.nan_to_num(matrix, posinf=0.0, neginf=0.0)
    
    # Standardization
    scaler = None
    if standardize:
        logger.info("  Standardizing features (zero mean, unit variance)...")
        scaler = StandardScaler()
        matrix = scaler.fit_transform(matrix)
        logger.info("  ✓ Data standardized")
    else:
        logger.info("  Using raw data (no standardization)")
    
    return matrix, sample_ids, unitig_ids, scaler


def load_metadata(sample_ids):
    """Load metadata for samples with explicit order preservation."""
    logger.info("Loading metadata...")
    
    # Load all metadata files
    train_meta = pl.read_csv("paper/metadata/train_metadata.tsv", separator="\t")
    test_meta = pl.read_csv("paper/metadata/test_metadata.tsv", separator="\t")
    
    # Combine and filter to our samples
    all_meta = pl.concat([train_meta, test_meta])
    metadata = all_meta.filter(pl.col("Run_accession").is_in(sample_ids))
    
    # CRITICAL: Reorder metadata to match sample_ids order exactly.
    # Some samples in the matrix (kmtricks.fof) may lack metadata entries; keep only
    # those that appear in both, preserving the matrix row order.
    meta_accessions = set(metadata['Run_accession'].to_list())
    filtered_sample_ids = [sid for sid in sample_ids if sid in meta_accessions]
    sample_to_idx = {sid: i for i, sid in enumerate(filtered_sample_ids)}
    
    # Convert to pandas for easier ordering, then back to polars
    metadata_pd = metadata.to_pandas()
    metadata_pd['_order'] = metadata_pd['Run_accession'].map(sample_to_idx)
    metadata_pd = metadata_pd.sort_values('_order').drop('_order', axis=1)
    metadata = pl.from_pandas(metadata_pd)
    
    # Verify order matches the (filtered) sample list
    if metadata['Run_accession'].to_list() != filtered_sample_ids:
        raise ValueError("Metadata order does not match sample_ids order!")
    
    if len(filtered_sample_ids) < len(sample_ids):
        n_skipped = len(sample_ids) - len(filtered_sample_ids)
        logger.warning(f"  Skipped {n_skipped} matrix samples with no metadata (not in train/test splits)")
    
    logger.info(f"  ✓ Loaded metadata for {len(metadata)} samples")
    logger.info(f"  ✓ Verified metadata order matches matrix rows")

    # Return the integer row indices into the original sample_ids list so the caller
    # can align embedding matrices (PCA/UMAP/t-SNE) to this metadata.
    meta_set = set(filtered_sample_ids)
    indices = [i for i, sid in enumerate(sample_ids) if sid in meta_set]
    return metadata, indices


def load_blast_annotations(unitig_ids):
    """Load BLAST annotations for unitigs with taxids from raw BLAST results."""
    logger.info("\nLoading BLAST annotations...")
    
    # Load raw BLAST results with taxids (field 13)
    raw_blast_path = Path('results/feature_analysis/all_features_blast/blast_results.txt')
    
    if not raw_blast_path.exists():
        logger.warning(f"  Raw BLAST results not found: {raw_blast_path}")
        # Fallback to processed file without taxids
        blast_path = Path(PATHS['feature_importance_dir']) / "unitigs_with_blast_hits.tsv"
        if not blast_path.exists():
            logger.warning("  No BLAST annotations found - skipping")
            return None
        logger.info(f"  Using processed file (no taxids): {blast_path}")
        return load_blast_annotations_fallback(blast_path, unitig_ids)
    
    logger.info(f"  Loading raw BLAST results with taxids: {raw_blast_path}")
    
    try:
        # Read raw BLAST format: qseqid sseqid pident length ... evalue bitscore staxids stitle
        # Columns: unitig_id, subject_id, pident, length, mismatch, gapopen, qstart, qend, 
        #          sstart, send, evalue, bitscore, taxid, description
        blast_df = pl.read_csv(
            raw_blast_path,
            separator="\t",
            has_header=False,
            new_columns=[
                'unitig_id', 'subject_id', 'pident', 'length', 'mismatch', 'gapopen',
                'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'taxid', 'description'
            ],
            schema_overrides={'taxid': pl.Utf8, 'unitig_id': pl.Int64}  # taxid as string, unitig_id as int
        )
        
        logger.info(f"  Raw BLAST file: {len(blast_df):,} rows")
        logger.info(f"  Sample taxids: {blast_df['taxid'].head(5).to_list()}")
        
        # Check ID ranges match
        blast_ids = blast_df['unitig_id'].unique().to_list()
        logger.info(f"  BLAST unitig ID range: {min(blast_ids):,} to {max(blast_ids):,}")
        logger.info(f"  Matrix unitig ID range: {unitig_ids.min():,} to {unitig_ids.max():,}")
        
        # Check overlap
        blast_ids_set = set(blast_ids)
        matrix_ids_set = set(unitig_ids.tolist())
        overlap = blast_ids_set & matrix_ids_set
        
        logger.info(f"  Unitigs in BLAST file: {len(blast_ids_set):,}")
        logger.info(f"  Unitigs in matrix: {len(matrix_ids_set):,}")
        logger.info(f"  Overlap (matching IDs): {len(overlap):,}")
        
        if len(overlap) == 0:
            logger.error(f"  CRITICAL: No matching unitig IDs between BLAST and matrix!")
            logger.error(f"  Sample BLAST IDs: {sorted(blast_ids)[:5]}")
            logger.error(f"  Sample matrix IDs: {sorted(unitig_ids.tolist())[:5]}")
            return None
        
        overlap_pct = len(overlap) / len(matrix_ids_set) * 100
        logger.info(f"  ✓ {overlap_pct:.1f}% of matrix unitigs have BLAST annotations")
        
        # Filter to only unitigs in our matrix and select best hit per unitig
        blast_df = blast_df.filter(pl.col("unitig_id").is_in(unitig_ids.tolist()))
        
        # Sort by bitscore (descending) and take best hit per unitig
        blast_df = (
            blast_df
            .sort('bitscore', descending=True)
            .group_by('unitig_id')
            .first()
        )
        
        logger.info(f"  ✓ Filtered to {len(blast_df):,} best BLAST hits for matrix unitigs")
        logger.info(f"  ✓ Taxids available for taxonomy resolution")
        return blast_df
    except Exception as e:
        logger.error(f"  Failed to load BLAST annotations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_blast_annotations_fallback(blast_path, unitig_ids):
    """Fallback loader for processed BLAST file without taxids."""
    try:
        blast_df = pl.read_csv(
            blast_path, 
            separator="\t",
            infer_schema_length=20000,
            schema_overrides={'blast_subject_length': pl.Utf8}
        )
        
        if 'unitig_id' not in blast_df.columns:
            logger.error(f"  'unitig_id' column not found")
            return None
        
        blast_df = blast_df.filter(pl.col("unitig_id").is_in(unitig_ids.tolist()))
        logger.info(f"  ✓ Loaded {len(blast_df):,} BLAST hits (no taxids)")
        return blast_df
    except Exception as e:
        logger.error(f"Fallback load failed: {e}")
        return None


def get_taxonomy_from_taxids(taxid_df, use_taxonkit=True):
    """
    Get full taxonomy lineages from NCBI taxonomy IDs using taxonkit.
    
    Args:
        taxid_df: DataFrame with 'unitig_id' and 'taxid' columns
        use_taxonkit: Whether to use taxonkit (if False, returns None)
    
    Returns:
        DataFrame with taxonomy_lineage column added, or None if fails
    """
    if not use_taxonkit:
        return None
    
    logger.info("  Resolving taxonomy from taxids using taxonkit...")
    
    # Check for taxonkit
    module_check_cmd = 'bash -c "type module 2>&1 && module load taxonkit/ 2>&1 && which taxonkit 2>&1"'
    
    try:
        result = subprocess.run(
            module_check_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0 or 'taxonkit' not in result.stdout:
            logger.warning("  taxonkit not available via module system")
            return None
        
        taxonkit_path = result.stdout.strip().split('\n')[-1]
        logger.info(f"  Found taxonkit: {taxonkit_path}")
        
    except Exception as e:
        logger.warning(f"  Could not check for taxonkit: {e}")
        return None
    
    # Convert to pandas for easier manipulation
    if isinstance(taxid_df, pl.DataFrame):
        taxid_df = taxid_df.to_pandas()
    
    # Filter out missing taxids
    valid_df = taxid_df[taxid_df['taxid'].notna()].copy()
    
    if len(valid_df) == 0:
        logger.warning("  No valid taxids to resolve")
        return None
    
    logger.info(f"  Resolving {len(valid_df)} unique taxids...")
    
    temp_input = None
    
    try:
        # Create temporary file with taxids
        import tempfile
        temp_input = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        
        # Write taxids (one per line)
        for taxid in valid_df['taxid'].unique():
            temp_input.write(f"{taxid}\n")
        temp_input.close()
        
        # Run taxonkit: lineage -> use field 2 (original lineage)
        # Get number of threads from environment
        import os
        n_threads = os.environ.get('OMP_NUM_THREADS', '16')
        
        cmd = (
            f'bash -c "module load taxonkit/ && '
            f'cat {temp_input.name} | '
            f'taxonkit lineage -i 1 -j {n_threads} '
            f'2>/dev/null"'
        )
        
        logger.info(f"  Running taxonkit with {n_threads} threads...")
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode != 0:
            logger.warning(f"  taxonkit failed with code {result.returncode}")
            return None
        
        # Parse output (format: taxid<tab>lineage)
        taxid_to_lineage = {}
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            fields = line.split('\t')
            if len(fields) >= 2:
                taxid = fields[0]
                lineage = fields[1]  # Original lineage
                
                # Remove "cellular organisms;" prefix if present
                if lineage.startswith('cellular organisms;'):
                    lineage = lineage.replace('cellular organisms;', '', 1)
                
                taxid_to_lineage[taxid] = lineage
        
        logger.info(f"  ✓ Resolved {len(taxid_to_lineage)} taxids")
        
        # Map back to dataframe
        valid_df['taxonomy_lineage'] = valid_df['taxid'].map(taxid_to_lineage)
        
        # Count successful resolutions
        success_count = valid_df['taxonomy_lineage'].notna().sum()
        logger.info(f"  ✓ {success_count}/{len(valid_df)} taxids have lineages")
        
        return valid_df
        
    except subprocess.TimeoutExpired:
        logger.warning("  taxonkit timed out after 5 minutes")
        return None
    except Exception as e:
        logger.warning(f"  Error running taxonkit: {e}")
        return None
    finally:
        # Cleanup temp files
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)


def perform_pca(matrix, n_components=50):
    """Perform PCA on unitig matrix."""
    logger.info(f"\nPerforming PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(matrix)
    
    logger.info(f"  ✓ PCA result shape: {pca_result.shape}")
    
    # Variance analysis
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    for threshold in [0.8, 0.9, 0.95]:
        # Find first index where cumsum >= threshold
        idx = np.argmax(cumsum_var >= threshold)
        # Handle edge case: if no PC meets threshold, use all PCs
        if cumsum_var[idx] < threshold and idx == 0:
            n_pcs = len(cumsum_var)
            actual_var = cumsum_var[-1]
        else:
            n_pcs = idx + 1
            actual_var = cumsum_var[n_pcs - 1]
        logger.info(f"  {n_pcs} PCs needed for ≥{threshold*100:.0f}% variance (actual: {actual_var*100:.2f}%)")
    
    variance_10 = pca.explained_variance_ratio_[:10].sum() * 100
    variance_50 = cumsum_var[-1] * 100 if len(cumsum_var) >= 50 else cumsum_var[-1] * 100
    logger.info(f"  First 10 PCs: {variance_10:.2f}% variance")
    logger.info(f"  All {n_components} PCs: {variance_50:.2f}% variance")
    
    return pca, pca_result


def perform_umap(matrix, n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1):
    """Perform UMAP dimensionality reduction (non-linear, better for complex structure)."""
    if not UMAP_AVAILABLE:
        logger.warning("\nUMAP not available - install with: pip install umap-learn")
        logger.warning("  Skipping UMAP analysis")
        return None
    
    logger.info(f"\nPerforming UMAP with {n_components} components...")
    logger.info(f"  n_neighbors={n_neighbors}, min_dist={min_dist}")
    if n_jobs == -1:
        logger.info(f"  Using all available CPU cores for parallel processing")
    else:
        logger.info(f"  Using {n_jobs} CPU cores")
    
    try:
        umap_model = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False,
            n_jobs=n_jobs
        )
        umap_result = umap_model.fit_transform(matrix)
        logger.info(f"  ✓ UMAP result shape: {umap_result.shape}")
        return umap_model, umap_result
    except Exception as e:
        logger.error(f"  Failed to perform UMAP: {e}")
        return None


def perform_tsne(matrix, n_components=2, perplexity=30):
    """Perform t-SNE dimensionality reduction (local structure preservation)."""
    logger.info(f"\nPerforming t-SNE with {n_components} components...")
    logger.info(f"  perplexity={perplexity}")
    
    if len(matrix) > 5000:
        logger.warning(f"  Warning: t-SNE can be slow for {len(matrix)} samples")
    
    try:
        tsne_model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            verbose=0
        )
        tsne_result = tsne_model.fit_transform(matrix)
        logger.info(f"  ✓ t-SNE result shape: {tsne_result.shape}")
        return tsne_result
    except Exception as e:
        logger.error(f"  Failed to perform t-SNE: {e}")
        return None


def calculate_separation_metrics(pca_result, metadata, n_pcs=10):
    """Calculate how well PCA separates different task classes."""
    logger.info("\nCalculating separation quality metrics...")
    
    metrics = {}
    
    for task in TASKS:
        if task not in metadata.columns:
            continue
            
        logger.info(f"  Task: {task}")
        labels = metadata[task].to_numpy()
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            logger.warning(f"    Skipping - only one class")
            continue
        
        # Silhouette score (using first n_pcs)
        pca_subset = pca_result[:, :n_pcs]
        
        try:
            silhouette = silhouette_score(pca_subset, labels, metric='euclidean')
            logger.info(f"    Silhouette score: {silhouette:.3f}")
            
            # Between-class vs within-class variance
            class_means = []
            class_vars = []
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    class_data = pca_subset[mask]
                    class_means.append(class_data.mean(axis=0))
                    class_vars.append(class_data.var(axis=0).mean())
            
            between_var = np.var(class_means, axis=0).mean()
            within_var = np.mean(class_vars)
            separation_ratio = between_var / (within_var + 1e-10)
            
            logger.info(f"    Separation ratio: {separation_ratio:.3f}")
            
            metrics[task] = {
                'silhouette_score': float(silhouette),
                'separation_ratio': float(separation_ratio),
                'n_classes': len(unique_labels),
                'n_pcs_used': n_pcs
            }
        except Exception as e:
            logger.error(f"    Failed to calculate metrics: {e}")
            metrics[task] = {'error': str(e)}
    
    return metrics


def calculate_pc_task_correlations(pca_result, metadata, pca_model, n_pcs=10):
    """Calculate ANOVA F-statistics to see which PCs correlate with which tasks."""
    logger.info("\nCalculating PC-task correlations...")
    
    correlations = {}
    
    for task in TASKS:
        if task not in metadata.columns:
            continue
            
        labels = metadata[task].to_numpy()
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            continue
        
        # ANOVA F-statistic for each PC
        f_stats = []
        p_values = []
        
        for pc_idx in range(min(n_pcs, pca_result.shape[1])):
            pc_values = pca_result[:, pc_idx]
            groups = [pc_values[labels == label] for label in unique_labels]
            
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                f_stats.append(float(f_stat))
                p_values.append(float(p_val))
            except:
                f_stats.append(0.0)
                p_values.append(1.0)
        
        # Find which PCs are most associated with this task
        top_pc_idx = np.argmax(f_stats)
        variance_explained = pca_model.explained_variance_ratio_[top_pc_idx] * 100
        
        logger.info(f"  {task}:")
        logger.info(f"    Strongest PC: PC{top_pc_idx + 1} (F={f_stats[top_pc_idx]:.2f}, p={p_values[top_pc_idx]:.2e})")
        logger.info(f"    PC{top_pc_idx + 1} explains {variance_explained:.2f}% variance")
        
        correlations[task] = {
            'f_statistics': f_stats,
            'p_values': p_values,
            'strongest_pc': int(top_pc_idx + 1),
            'strongest_f_stat': float(f_stats[top_pc_idx]),
            'strongest_p_value': float(p_values[top_pc_idx])
        }
    
    return correlations


def save_pca_model(pca, pca_result, sample_ids, metadata, unitig_ids, scaler=None, umap_model=None, umap_result=None):
    """Save PCA model and reference data for future projections."""
    logger.info("\nSaving PCA model for future projections...")
    
    output_path = Path("models/pca_reference.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PCA model and reference information
    reference_data = {
        'pca_model': pca,
        'pca_coordinates': pca_result,  # Save pre-computed reference PCA coords
        'umap_model': umap_model,  # Save UMAP model for projecting new samples
        'umap_coordinates': umap_result,  # Save pre-computed reference UMAP coords
        'scaler': scaler,  # Save scaler for standardization (None if not standardized)
        'sample_ids': sample_ids,
        'unitig_ids': unitig_ids.tolist(),
        'metadata': metadata.to_dict(),
        'n_features': pca.n_features_in_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'standardized': scaler is not None,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(reference_data, f)
    
    logger.info(f"  ✓ Saved to {output_path}")
    logger.info(f"  Model expects {pca.n_features_in_:,} features (unitigs)")
    if scaler is not None:
        logger.info(f"  Data was standardized (scaler saved for new samples)")
    else:
        logger.info(f"  Data was not standardized (raw values used)")
    if umap_model is not None:
        logger.info(f"  UMAP model saved for projecting new samples")
    return output_path


def export_pca_coordinates(pca_result, sample_ids, metadata, pca_model, output_dir):
    """Export PCA coordinates to CSV with metadata."""
    logger.info("\nExporting PCA coordinates...")
    
    # Create DataFrame with PC coordinates
    n_pcs = min(10, pca_result.shape[1])  # Export first 10 PCs
    pc_cols = {f'PC{i+1}': pca_result[:, i] for i in range(n_pcs)}
    
    df = pd.DataFrame({
        'sample_id': sample_ids,
        **pc_cols,
    })
    
    # Add metadata
    meta_pd = metadata.to_pandas()
    df = df.merge(meta_pd[['Run_accession'] + TASKS], 
                  left_on='sample_id', right_on='Run_accession', how='left')
    
    # Add variance explained as comment in header
    output_path = output_dir / 'pca_coordinates.csv'
    with open(output_path, 'w') as f:
        f.write(f"# PCA coordinates for {len(sample_ids)} samples\n")
        f.write(f"# Variance explained by each PC:\n")
        for i in range(n_pcs):
            var_pct = pca_model.explained_variance_ratio_[i] * 100
            f.write(f"# PC{i+1}: {var_pct:.2f}%\n")
        df.to_csv(f, index=False)
    
    logger.info(f"  ✓ Saved {len(df)} samples to {output_path}")
    return output_path


def export_embedding_coordinates(embedding_result, sample_ids, metadata, method_name, output_dir):
    """Export UMAP/t-SNE coordinates to CSV with metadata."""
    logger.info(f"\nExporting {method_name} coordinates...")
    
    # Create DataFrame with embedding coordinates
    df = pd.DataFrame({
        'sample_id': sample_ids,
        f'{method_name}_dim1': embedding_result[:, 0],
        f'{method_name}_dim2': embedding_result[:, 1],
    })
    
    # Add metadata
    meta_pd = metadata.to_pandas()
    df = df.merge(meta_pd[['Run_accession'] + TASKS], 
                  left_on='sample_id', right_on='Run_accession', how='left')
    
    # Save to CSV
    output_path = output_dir / f'{method_name.lower()}_coordinates.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"  ✓ Saved {len(df)} samples to {output_path}")
    return output_path


def export_pca_loadings(pca_model, unitig_ids, output_dir, top_n=100):
    """Export PCA loadings (feature contributions) to CSV."""
    logger.info(f"\nExporting PCA loadings (top {top_n} per PC)...")
    
    # Get loadings for first 10 PCs
    n_pcs = min(10, pca_model.n_components_)
    loadings = pca_model.components_[:n_pcs, :].T  # Features × PCs
    
    # Create DataFrame
    loading_cols = {f'PC{i+1}_loading': loadings[:, i] for i in range(n_pcs)}
    df = pd.DataFrame({
        'unitig_id': unitig_ids,
        **loading_cols,
    })
    
    # Add absolute max loading across all PCs
    df['max_abs_loading'] = np.abs(loadings).max(axis=1)
    
    # Save full loadings
    full_path = output_dir / 'pca_loadings_all.csv'
    df.to_csv(full_path, index=False)
    logger.info(f"  ✓ Saved all {len(df):,} unitig loadings to {full_path}")
    
    # Save top unitigs by loading
    top_df = df.nlargest(top_n, 'max_abs_loading')
    top_path = output_dir / f'pca_loadings_top{top_n}.csv'
    top_df.to_csv(top_path, index=False)
    logger.info(f"  ✓ Saved top {top_n} unitigs to {top_path}")
    
    return full_path, top_path


def plot_pca_by_task(pca_result, metadata, task_col, pca_model, output_prefix):
    """Create PCA plots colored by task labels using PLOT_CONFIG palette."""
    logger.info(f"\nPlotting PCA for {task_col}...")
    
    # Prepare data
    df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Run_accession': metadata['Run_accession'].to_list(),
        task_col: metadata[task_col].to_list(),
    })
    
    # Get variance explained for axis labels
    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    xlabel = f'PC1 ({var_pc1:.1f}%)'
    ylabel = f'PC2 ({var_pc2:.1f}%)'
    
    # Get unique classes and assign colors from PLOT_CONFIG
    unique_classes = sorted(df[task_col].unique())
    color_palette = PLOT_CONFIG['colors']['palette']
    color_map = {cls: color_palette[i % len(color_palette)] 
                 for i, cls in enumerate(unique_classes)}
    
    # Create Plotly figure
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color=task_col,
        color_discrete_map=color_map,
        hover_data=['Run_accession'],
        title=f'PCA of Unitig Matrix - {task_col.replace("_", " ").title()}',
        labels={'PC1': xlabel, 'PC2': ylabel},
        template=PLOT_CONFIG['template'],
        width=PLOT_CONFIG['sizes']['default_width'],
        height=PLOT_CONFIG['sizes']['default_height']
    )
    
    fig.update_traces(
        marker=dict(
            size=6,
            opacity=PLOT_CONFIG['marker_opacity'],
            line=dict(width=0.5, color='white')
        )
    )
    fig.update_layout(
        font=dict(size=PLOT_CONFIG['font_size']),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Save interactive HTML
    html_path = Path(PATHS['figures_dir']) / f"{output_prefix}_{task_col}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved interactive HTML: {html_path}")
    
    # Save static PNG using Plotly
    png_path = Path(PATHS['figures_dir']) / f"{output_prefix}_{task_col}.png"
    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        logger.info(f"  ✓ Saved static PNG: {png_path}")
    except Exception as e:
        logger.warning(f"  Could not save PNG (kaleido not installed?): {e}")
        logger.info(f"  Install with: pip install -U kaleido")


def plot_embedding_by_task(embedding_result, metadata, task_col, method_name, output_prefix, var_labels=None):
    """Generic plotting function for any dimensionality reduction method (UMAP, t-SNE, etc.)."""
    logger.info(f"\nPlotting {method_name} for {task_col}...")
    
    # Prepare data
    df = pd.DataFrame({
        'Dim1': embedding_result[:, 0],
        'Dim2': embedding_result[:, 1],
        'Run_accession': metadata['Run_accession'].to_list(),
        task_col: metadata[task_col].to_list(),
    })
    
    # Set axis labels
    if var_labels:
        xlabel, ylabel = var_labels
    else:
        xlabel = f'{method_name} Dimension 1'
        ylabel = f'{method_name} Dimension 2'
    
    # Get unique classes and assign colors from PLOT_CONFIG
    unique_classes = sorted(df[task_col].unique())
    color_palette = PLOT_CONFIG['colors']['palette']
    color_map = {cls: color_palette[i % len(color_palette)] 
                 for i, cls in enumerate(unique_classes)}
    
    # Create Plotly figure
    fig = px.scatter(
        df,
        x='Dim1',
        y='Dim2',
        color=task_col,
        color_discrete_map=color_map,
        hover_data=['Run_accession'],
        title=f'{method_name} of Unitig Matrix - {task_col.replace("_", " ").title()}',
        labels={'Dim1': xlabel, 'Dim2': ylabel},
        template=PLOT_CONFIG['template'],
        width=PLOT_CONFIG['sizes']['default_width'],
        height=PLOT_CONFIG['sizes']['default_height']
    )
    
    fig.update_traces(
        marker=dict(
            size=6,
            opacity=PLOT_CONFIG['marker_opacity'],
            line=dict(width=0.5, color='white')
        )
    )
    fig.update_layout(
        font=dict(size=PLOT_CONFIG['font_size']),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Save interactive HTML
    html_path = Path(PATHS['figures_dir']) / f"{output_prefix}_{task_col}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved interactive HTML: {html_path}")
    
    # Save static PNG using Plotly
    png_path = Path(PATHS['figures_dir']) / f"{output_prefix}_{task_col}.png"
    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        logger.info(f"  ✓ Saved static PNG: {png_path}")
    except Exception as e:
        logger.warning(f"  Could not save PNG: {e}")


def plot_scree_plot(pca, output_path):
    """Create scree plot showing explained variance (plotly)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    logger.info("\nCreating scree plot...")

    evr = pca.explained_variance_ratio_
    n_components = len(evr)
    pcs = list(range(1, n_components + 1))
    cumsum = list(np.cumsum(evr))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Explained Variance by Principal Component',
            'Cumulative Explained Variance'
        ]
    )

    # Panel 1: bar chart of individual explained variance
    fig.add_trace(go.Bar(
        x=pcs, y=list(evr),
        marker_color='steelblue', opacity=0.7,
        showlegend=False
    ), row=1, col=1)

    # Panel 2: cumulative line
    fig.add_trace(go.Scatter(
        x=pcs, y=cumsum,
        mode='lines+markers', line=dict(color='steelblue', width=2),
        name='Cumulative variance'
    ), row=1, col=2)
    fig.add_hline(y=0.8, line=dict(color='red', dash='dash'), annotation_text='80%', row=1, col=2)
    fig.add_hline(y=0.9, line=dict(color='orange', dash='dash'), annotation_text='90%', row=1, col=2)

    fig.update_xaxes(title_text='Principal Component', row=1, col=1)
    fig.update_yaxes(title_text='Explained Variance Ratio', row=1, col=1)
    fig.update_xaxes(title_text='Number of Principal Components', row=1, col=2)
    fig.update_yaxes(title_text='Cumulative Explained Variance', row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        width=1400, height=500,
        font=dict(size=13)
    )

    output_path = Path(output_path)
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    fig.write_image(str(output_path), width=1400, height=500, scale=2)
    logger.info(f"  ✓ Saved scree plot: {output_path}")


def extract_taxonomy_category(taxonomy_string, level='phylum', fallback_value=None):
    """
    Extract taxonomy category from BLAST taxonomy string.
    
    Taxonomy formats can vary:
    - "Bacteria; Proteobacteria; Gammaproteobacteria; Enterobacterales; Enterobacteriaceae; Escherichia; Escherichia coli"
    - "cellular organisms; Bacteria; Proteobacteria..."
    - "Viruses; Duplodnaviria; Heunggongvirae..."
    
    Args:
        taxonomy_string: Full taxonomy string
        level: 'kingdom', 'phylum', 'class', or 'order'
        fallback_value: Value to return if extraction fails (instead of 'Unknown')
    
    Returns:
        Extracted category or fallback_value or 'Unknown'
    """
    if pd.isna(taxonomy_string) or not taxonomy_string:
        return fallback_value if fallback_value else 'Unknown'
    
    parts = [p.strip() for p in str(taxonomy_string).split(';')]
    
    # Common kingdoms/domains at start
    kingdoms = ['Bacteria', 'Archaea', 'Eukaryota', 'Viruses', 'Viridiplantae', 'Metazoa', 'Fungi']
    
    if level == 'kingdom':
        for part in parts[:3]:  # Check first few levels
            if part in kingdoms:
                return part
        # If no match, use first non-generic term
        for part in parts:
            if part not in ['cellular organisms', 'root', '']:
                return part
        return fallback_value if fallback_value else 'Unknown'
    
    elif level == 'phylum':
        # Phylum is usually 2nd or 3rd level after kingdom
        if len(parts) >= 2:
            # Skip generic terms
            for part in parts[1:4]:
                if part not in kingdoms + ['cellular organisms', 'root', '']:
                    return part
        return fallback_value if fallback_value else 'Unknown'
    
    elif level == 'class':
        # Class is usually 3rd-4th level
        if len(parts) >= 3:
            for part in parts[2:5]:
                if part not in kingdoms + ['cellular organisms', 'root', '']:
                    return part
        return fallback_value if fallback_value else 'Unknown'
    
    elif level == 'order':
        # Order is usually 4th-5th level
        if len(parts) >= 4:
            for part in parts[3:6]:
                if part not in kingdoms + ['cellular organisms', 'root', '']:
                    return part
        return fallback_value if fallback_value else 'Unknown'
    
    return fallback_value if fallback_value else 'Unknown'


def extract_genus_from_species(species_name):
    """
    Extract genus from species name, handling edge cases.
    
    Handles:
    - "Genus species" → "Genus"
    - "Candidatus Genus species" → "Genus"
    - "[Genus] species" → "Genus"
    - "uncultured bacterium" → "uncultured bacterium"
    - Single word → return as-is
    """
    if pd.isna(species_name) or not species_name:
        return "Unknown"
    
    name = str(species_name).strip()
    
    # Remove brackets
    name = name.replace('[', '').replace(']', '')
    
    # Split into words
    words = name.split()
    
    if len(words) == 0:
        return "Unknown"
    
    # Handle "Candidatus Genus species" → skip "Candidatus"
    if words[0].lower() == 'candidatus' and len(words) > 1:
        return words[1]
    
    # Handle "uncultured bacterium/organism" → keep both words for context
    if words[0].lower() in ['uncultured', 'unclassified', 'unidentified'] and len(words) > 1:
        return f"{words[0]} {words[1]}"
    
    # Standard case: first word is genus
    return words[0]


def get_taxonomy_from_species(species_names_df, use_taxonkit=True):
    """
    Get full taxonomy lineages from species names using taxonkit or fallback to genus.
    
    Args:
        species_names_df: DataFrame with 'unitig_id' and species name column
        use_taxonkit: Try to use taxonkit for proper taxonomy (recommended)
    
    Returns:
        DataFrame with 'unitig_id', 'species', and 'taxonomy_lineage' columns
    """
    import subprocess
    import tempfile
    
    if not use_taxonkit:
        logger.info("  Skipping taxonkit - using genus extraction only")
        return None
    
    logger.info("  Attempting to get taxonomy lineages using taxonkit...")
    
    # Check cache first
    cache_path = Path(PATHS['feature_importance_dir']) / 'taxonkit_cache.pkl'
    if cache_path.exists():
        logger.info(f"  ✓ Loading cached taxonomy from {cache_path}")
        try:
            import pickle
            with open(cache_path, 'rb') as f:
                taxonomy_map = pickle.load(f)
            logger.info(f"  ✓ Loaded {len(taxonomy_map)} cached taxonomy entries")
            
            # Add to dataframe
            species_names_df['taxonomy_lineage'] = species_names_df['species'].map(taxonomy_map)
            return species_names_df
        except Exception as e:
            logger.warning(f"  Cache loading failed: {e}, will regenerate...")
    
    # Check if taxonkit is available via module system
    logger.info("  Checking for taxonkit availability...")
    try:
        # Test if module command exists and taxonkit module is available
        check_result = subprocess.run(
            'bash -c "type module && module load taxonkit/ 2>&1 && which taxonkit"',
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if check_result.returncode != 0:
            logger.warning("  taxonkit module not available")
            logger.warning("  Load with: module load taxonkit/")
            return None
            
        taxonkit_path = check_result.stdout.strip().split('\n')[-1]
        logger.info(f"  ✓ Found taxonkit: {taxonkit_path}")
        
    except Exception as e:
        logger.warning(f"  Failed to check for taxonkit: {e}")
        return None
    
    # Get number of threads to use (from environment or default to 16)
    n_threads = int(os.environ.get('OMP_NUM_THREADS', 16))
    
    species_file = None
    try:
        # Create temp file with species names
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            species_file = f.name
            unique_species = species_names_df['species'].unique()
            for species in unique_species:
                if pd.notna(species) and str(species).strip():
                    f.write(f"{species}\n")
        
        logger.info(f"  Running taxonkit on {len(unique_species)} unique species names (using {n_threads} threads)...")
        
        # FIXED: Pipeline outputs are: Field1=species, Field2=taxid, Field3=lineage
        # Use reformat2 -I 2 (taxid field) for cleaner approach
        taxonkit_cmd = (
            f"module load taxonkit/ && "
            f"cat {species_file} | "
            f"taxonkit name2taxid -i 1 -j {n_threads} 2>/dev/null | "
            f"taxonkit lineage -i 2 -j {n_threads} 2>/dev/null | "
            f"taxonkit reformat2 -I 2 -j {n_threads} "
            f"-f '{{domain|acellular root|superkingdom}};{{phylum}};{{class}};{{order}};{{family}};{{genus}};{{species}}' "
            f"2>/dev/null"
        )
        
        logger.info(f"  Command: {taxonkit_cmd[:100]}...")
        
        result = subprocess.run(
            taxonkit_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logger.warning(f"  taxonkit failed with return code {result.returncode}")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr[:200]}")
            return None
        
        # Parse results - format is: species_name \t taxid \t original_lineage \t reformatted_lineage
        taxonomy_map = {}
        lines_parsed = 0
        lines_failed = 0
        
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            
            parts = line.split('\t')
            # reformat2 adds a 4th field with reformatted lineage
            if len(parts) >= 4:
                species = parts[0].strip()
                lineage = parts[3].strip()  # Field 4 = reformatted lineage
                
                # Only add if lineage is not empty
                if lineage:
                    taxonomy_map[species] = lineage
                    lines_parsed += 1
                else:
                    lines_failed += 1
            elif len(parts) >= 3:
                # Fallback for old format (if reformat2 didn't add field)
                species = parts[0].strip()
                lineage = parts[2].strip()
                if lineage:
                    taxonomy_map[species] = lineage
                    lines_parsed += 1
                else:
                    lines_failed += 1
            else:
                lines_failed += 1
        
        logger.info(f"  ✓ Parsed {lines_parsed} taxonomy lineages")
        if lines_failed > 0:
            logger.warning(f"  ⚠ Failed to parse {lines_failed} lines")
        
        if len(taxonomy_map) == 0:
            logger.warning("  No taxonomy data retrieved - taxonkit may have failed")
            return None
        
        logger.info(f"  ✓ Got taxonomy for {len(taxonomy_map)}/{len(unique_species)} species using taxonkit")
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(taxonomy_map, f)
        logger.info(f"  ✓ Saved taxonomy cache to {cache_path}")
        
        # Add to dataframe
        species_names_df['taxonomy_lineage'] = species_names_df['species'].map(taxonomy_map)
        
        return species_names_df
        
    except subprocess.TimeoutExpired:
        logger.warning("  taxonkit command timed out after 300 seconds")
        return None
    except Exception as e:
        logger.warning(f"  taxonkit failed: {e}")
        import traceback
        logger.warning(f"  Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Clean up temp file
        if species_file and os.path.exists(species_file):
            try:
                os.unlink(species_file)
            except:
                pass


def load_feature_importance(unitig_ids):
    """Load feature importance scores from classification results if available."""
    logger.info("\nAttempting to load feature importance scores...")
    
    importance_files = [
        Path(PATHS['feature_importance_dir']) / 'feature_importance_summary.tsv',
        Path(PATHS['feature_importance_dir']) / 'top_features_by_task.tsv',
    ]
    
    for imp_file in importance_files:
        if imp_file.exists():
            try:
                imp_df = pl.read_csv(imp_file, separator='\t')
                if 'unitig_id' in imp_df.columns:
                    # Filter to our unitigs
                    imp_df = imp_df.filter(pl.col('unitig_id').is_in(unitig_ids.tolist()))
                    logger.info(f"  ✓ Loaded importance scores for {len(imp_df):,} unitigs from {imp_file.name}")
                    return imp_df
            except Exception as e:
                logger.warning(f"  Failed to load {imp_file.name}: {e}")
    
    logger.info("  No feature importance files found - will use loading magnitude only")
    return None


def plot_pca_loadings(pca_model, unitig_ids, blast_annotations, output_dir, 
                      feature_importance=None, color_by='phylum'):
    """Plot unitig loadings in PC1 vs PC2 space with BLAST annotations.
    
    Creates two plots:
    1. Random 1% sample - Random selection from ALL unitigs with BLAST hits
    2. Discriminant features - Top 100 features per task from classification model
    
    Args:
        pca_model: Fitted PCA model
        unitig_ids: Array of unitig IDs mapping feature_index → unitig_id
        blast_annotations: DataFrame with ALL BLAST hits (raw or processed, must have 'unitig_id' and ideally 'taxid')
        output_dir: Output directory for plots
        feature_importance: DataFrame with DISCRIMINANT features (from blast_annotations.tsv with 'feature_index' column)
        color_by: Taxonomy level for coloring ('kingdom', 'phylum', 'class', 'order')
    """
    logger.info(f"\nCreating PCA loading plots with BLAST-annotated unitigs...")
    logger.info(f"  Coloring by taxonomy level: {color_by}")
    logger.info(f"  Data sources:")
    logger.info(f"    - Plot 1 (random): blast_annotations (all BLAST hits)")
    logger.info(f"    - Plot 2 (discriminant): feature_importance (top features from model)")
    
    if blast_annotations is None or len(blast_annotations) == 0:
        logger.warning("  No BLAST annotations available - skipping loading plots")
        return
    
    # Get PC1 and PC2 loadings
    pc1_loadings = pca_model.components_[0, :]  # First PC
    pc2_loadings = pca_model.components_[1, :]  # Second PC
    
    # Get variance explained by each PC for weighting
    var_pc1 = pca_model.explained_variance_ratio_[0]
    var_pc2 = pca_model.explained_variance_ratio_[1]
    
    logger.info(f"  PC1 variance: {var_pc1*100:.2f}%, PC2 variance: {var_pc2*100:.2f}%")
    
    # Create DataFrame with loadings for ALL features
    loading_df = pd.DataFrame({
        'feature_index': np.arange(len(unitig_ids)),  # 0 to n_features-1
        'unitig_id': unitig_ids,
        'PC1_loading': pc1_loadings,
        'PC2_loading': pc2_loadings,
        'loading_magnitude': np.sqrt((pc1_loadings * var_pc1)**2 + (pc2_loadings * var_pc2)**2)
    })
    
    # Get BLAST data
    blast_df = blast_annotations.to_pandas()
    blast_unitigs = set(blast_df['unitig_id'].to_list())
    
    logger.info(f"  Total unitigs: {len(unitig_ids):,}")
    logger.info(f"  Unitigs with BLAST hits: {len(blast_unitigs):,}")
    
    # ========== PLOT 1: Random 1% sample of BLAST-annotated unitigs ==========
    logger.info(f"\n  Creating Plot 1: Random 1% sample of BLAST-annotated unitigs...")
    
    # Filter to unitigs with BLAST
    loading_blast = loading_df[loading_df['unitig_id'].isin(blast_unitigs)].copy()
    
    # Sample 1%
    n_sample = max(int(len(loading_blast) * 0.01), 100)  # At least 100
    sample_unitigs = loading_blast.sample(n=min(n_sample, len(loading_blast)), random_state=42)
    
    logger.info(f"    Sampled {len(sample_unitigs)} unitigs (1% of {len(loading_blast):,} BLAST hits)")
    
    # Add taxonomy annotations
    sample_with_tax = _add_taxonomy_annotations(
        sample_unitigs, blast_df, color_by, unitig_ids, logger
    )
    
    # Plot
    _plot_loadings_scatter(
        sample_with_tax, loading_blast, pca_model, color_by,
        output_dir / f'sup_04_pca_loadings_random1pct_{color_by}.html',
        output_dir / f'sup_04_pca_loadings_random1pct_{color_by}.png',
        f"Random 1% Sample ({len(sample_with_tax)} unitigs)",
        logger
    )
    
    # ========== PLOT 2: Top discriminant features per task ==========
    if feature_importance is not None and 'feature_index' in feature_importance.columns:
        logger.info(f"\n  Creating Plot 2: Top discriminant features per task...")
        
        # Load discriminant feature indices
        if isinstance(feature_importance, pl.DataFrame):
            feature_importance = feature_importance.to_pandas()
        
        discriminant_indices = feature_importance['feature_index'].unique()
        logger.info(f"    Found {len(discriminant_indices)} discriminant feature indices")
        
        # CRITICAL MAPPING: feature_index → unitig_id
        # feature_index is the matrix column position (0 to n_features-1)
        # unitig_ids array maps: unitig_ids[feature_index] = actual_unitig_id
        # Example: feature_index=0 → unitig_ids[0] = 2368143 (actual ID)
        discriminant_unitig_ids = [unitig_ids[idx] for idx in discriminant_indices if idx < len(unitig_ids)]
        
        logger.info(f"    Sample mapping: feature_index[0] → unitig_id {unitig_ids[0] if len(unitig_ids) > 0 else 'N/A'}")
        
        logger.info(f"    Mapped to {len(discriminant_unitig_ids)} unitig IDs")
        
        # Filter to those with BLAST annotations
        discriminant_with_blast = loading_df[
            loading_df['unitig_id'].isin(discriminant_unitig_ids) & 
            loading_df['unitig_id'].isin(blast_unitigs)
        ].copy()
        
        logger.info(f"    {len(discriminant_with_blast)} discriminant features have BLAST annotations")
        
        if len(discriminant_with_blast) > 0:
            # Add taxonomy annotations
            discriminant_with_tax = _add_taxonomy_annotations(
                discriminant_with_blast, blast_df, color_by, unitig_ids, logger
            )
            
            # Plot
            _plot_loadings_scatter(
                discriminant_with_tax, loading_blast, pca_model, color_by,
                output_dir / f'sup_04_pca_loadings_discriminant_{color_by}.html',
                output_dir / f'sup_04_pca_loadings_discriminant_{color_by}.png',
                f"Top Discriminant Features ({len(discriminant_with_tax)} unitigs)",
                logger
            )
        else:
            logger.warning("    No discriminant features with BLAST - skipping plot 2")
    else:
        logger.warning("  No feature importance data - skipping discriminant features plot")


def _add_taxonomy_annotations(unitigs_df, blast_df, color_by, unitig_ids, logger):
    """Helper to add taxonomy annotations to unitigs."""
    
    # Check if we have taxid column (from raw BLAST)
    if 'taxid' in blast_df.columns:
        logger.info("    Using taxids for taxonomy resolution")
        
        # Group by unitig_id, take first (best) hit
        blast_grouped = blast_df.groupby('unitig_id').first().reset_index()
        
        # Get taxonomy from taxids using taxonkit
        blast_with_tax = get_taxonomy_from_taxids(blast_grouped[['unitig_id', 'taxid']], use_taxonkit=True)
        
        if blast_with_tax is not None and 'taxonomy_lineage' in blast_with_tax.columns:
            blast_grouped = blast_grouped.merge(
                blast_with_tax[['unitig_id', 'taxonomy_lineage']], 
                on='unitig_id', 
                how='left'
            )
            
            # Extract category from lineage
            blast_grouped['taxonomy_category'] = blast_grouped['taxonomy_lineage'].apply(
                lambda x: extract_taxonomy_category(x, level=color_by, fallback_value='Unknown') if pd.notna(x) else 'Unknown'
            )
            
            logger.info(f"    ✓ Taxonomy resolved via taxids")
        else:
            logger.warning("    taxonkit failed - using description as fallback")
            blast_grouped['taxonomy_category'] = blast_grouped['description'].apply(
                lambda x: extract_genus_from_species(str(x)) if pd.notna(x) else 'Unknown'
            )
    else:
        logger.info("    No taxids - using species/description fallback")
        
        # Extract taxonomy column
        tax_col = None
        for col in ['taxonomy', 'blast_taxonomy', 'lineage', 'blast_species', 'species', 
                    'description', 'blast_description', 'best_hit_species']:
            if col in blast_df.columns:
                tax_col = col
                break
        
        if tax_col is None:
            logger.warning("    No taxonomy column found - using genus fallback")
            if 'genus' in blast_df.columns:
                blast_df['taxonomy_category'] = blast_df['genus'].fillna('Unknown')
            else:
                blast_df['taxonomy_category'] = 'No taxonomy'
            blast_grouped = blast_df.groupby('unitig_id')['taxonomy_category'].first().reset_index()
        else:
            # Group by unitig_id, take first hit
            blast_grouped = blast_df.groupby('unitig_id')[tax_col].first().reset_index()
            blast_grouped.rename(columns={tax_col: 'species'}, inplace=True)
            
            # Check if we have full taxonomy lineages
            sample_value = blast_grouped['species'].iloc[0] if len(blast_grouped) > 0 else ""
            has_full_taxonomy = ';' in str(sample_value)
            
            if not has_full_taxonomy:
                # Fallback to genus extraction
                blast_grouped['taxonomy_category'] = blast_grouped['species'].apply(extract_genus_from_species)
            else:
                # Already have full taxonomy
                blast_grouped['taxonomy_category'] = blast_grouped['species'].apply(
                    lambda x: extract_taxonomy_category(x, level=color_by, fallback_value=str(x)[:30])
                )
    
    # Merge with unitigs
    result = unitigs_df.merge(
        blast_grouped[['unitig_id', 'taxonomy_category']], 
        on='unitig_id', 
        how='left'
    )
    result['taxonomy_category'] = result['taxonomy_category'].fillna('Unknown')
    
    # Filter out "Unknown" ONLY if we have enough other data
    unknown_count = (result['taxonomy_category'] == 'Unknown').sum()
    known_count = len(result) - unknown_count
    
    if known_count > 10:  # Only filter if we have enough known taxonomy
        result = result[result['taxonomy_category'] != 'Unknown'].copy()
        logger.info(f"    Filtered out {unknown_count} 'Unknown' entries, kept {known_count} with taxonomy")
    else:
        logger.warning(f"    Only {known_count} unitigs with known taxonomy - keeping Unknown entries to avoid empty plot")
    
    logger.info(f"    Final dataset: {len(result)} unitigs with taxonomy at {color_by} level")
    
    return result


def _plot_loadings_scatter(unitigs_df, all_blast_df, pca_model, color_by, 
                           html_path, png_path, title_suffix, logger):
    """Helper to create loading scatter plot."""
    
    if len(unitigs_df) == 0:
        logger.warning(f"    No unitigs to plot - skipping")
        return
    
    logger.info(f"    Creating plot with {len(unitigs_df)} unitigs...")
    
    # Get unique categories
    categories = sorted(unitigs_df['taxonomy_category'].unique())
    n_colors = len(categories)
    
    logger.info(f"    Taxonomy categories: {n_colors}")
    category_dist = unitigs_df['taxonomy_category'].value_counts().head(10).to_dict()
    logger.info(f"    Top categories: {category_dist}")
    
    # Color palette
    if n_colors <= 10:
        colors = px.colors.qualitative.Plotly
    elif n_colors <= 24:
        colors = px.colors.qualitative.Dark24
    else:
        colors = px.colors.sample_colorscale('viridis', [i/n_colors for i in range(n_colors)])
    
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    # Create figure
    fig = go.Figure()
    
    # Add background (all BLAST unitigs)
    fig.add_trace(go.Scatter(
        x=all_blast_df['PC1_loading'],
        y=all_blast_df['PC2_loading'],
        mode='markers',
        marker=dict(size=2, color='lightgray', opacity=0.2),
        name='Other BLAST unitigs',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # Add colored unitigs by category
    for category in categories:
        cat_data = unitigs_df[unitigs_df['taxonomy_category'] == category]
        if len(cat_data) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=cat_data['PC1_loading'],
            y=cat_data['PC2_loading'],
            mode='markers',
            marker=dict(
                size=8,
                color=category_colors[category],
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            text=[f"ID:{uid}" for uid in cat_data['unitig_id']],
            name=f'{category} ({len(cat_data)})',
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>' + category + '<extra></extra>'
        ))
    
    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    
    fig.update_layout(
        title=f'PCA Loadings - {title_suffix} (by {color_by.capitalize()})',
        xaxis_title=f'PC1 Loading ({var_pc1:.1f}% variance)',
        yaxis_title=f'PC2 Loading ({var_pc2:.1f}% variance)',
        template=PLOT_CONFIG['template'],
        width=1400,
        height=1000,
        font=dict(size=PLOT_CONFIG['font_size']),
        hovermode='closest',
        legend=dict(
            title=f'{color_by.capitalize()}',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )
    
    # Save
    fig.write_html(str(html_path))
    logger.info(f"    ✓ Saved HTML: {html_path.name}")
    
    try:
        fig.write_image(str(png_path), width=1400, height=1000, scale=2)
        logger.info(f"    ✓ Saved PNG: {png_path.name}")
    except Exception as e:
        logger.warning(f"    Could not save PNG: {e}")


def plot_unitig_pca_by_top_species(pca_model, unitig_ids, blast_annotations, output_dir, top_n=10):
    """
    Plot ALL unitigs in PCA loading space, colored by top N most frequent species.
    
    Args:
        pca_model: Fitted PCA model
        unitig_ids: Array of unitig IDs
        blast_annotations: DataFrame with BLAST hits including 'taxid' column
        output_dir: Output directory for plots
        top_n: Number of top species to show (default: 10)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Creating unitig PCA loadings plot colored by top {top_n} species...")
    logger.info(f"{'='*80}")
    
    if blast_annotations is None or len(blast_annotations) == 0:
        logger.warning("  No BLAST annotations available - skipping species plot")
        return
    
    # Convert to pandas if needed
    if hasattr(blast_annotations, 'to_pandas'):
        blast_df = blast_annotations.to_pandas()
    else:
        blast_df = blast_annotations
    
    # Check for taxid column
    if 'taxid' not in blast_df.columns:
        logger.warning("  No taxid column - cannot create species plot")
        return
    
    # Get PC1 and PC2 loadings
    pc1_loadings = pca_model.components_[0, :]
    pc2_loadings = pca_model.components_[1, :]
    var_pc1 = pca_model.explained_variance_ratio_[0] * 100
    var_pc2 = pca_model.explained_variance_ratio_[1] * 100
    
    # Create loading DataFrame for ALL unitigs
    loading_df = pd.DataFrame({
        'unitig_id': unitig_ids,
        'PC1_loading': pc1_loadings,
        'PC2_loading': pc2_loadings
    })
    
    logger.info(f"  Total unitigs: {len(loading_df):,}")
    logger.info(f"  BLAST hits: {len(blast_df):,}")
    
    # Resolve taxids to get full lineages (including species)
    logger.info(f"  Resolving taxids to species names...")
    unique_taxids = blast_df['taxid'].dropna().unique()
    logger.info(f"    Found {len(unique_taxids)} unique taxids")
    
    taxid_to_lineage = {}
    temp_input = None
    
    try:
        import tempfile
        import subprocess
        import os
        
        temp_input = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        
        # Write taxids
        for taxid in unique_taxids:
            temp_input.write(f"{taxid}\n")
        temp_input.close()
        
        # Get threads
        n_threads = os.environ.get('OMP_NUM_THREADS', '16')
        
        # Run taxonkit to get full lineage
        cmd = (
            f'bash -c "module load taxonkit/ && '
            f'cat {temp_input.name} | '
            f'taxonkit lineage -i 1 -j {n_threads} '
            f'2>/dev/null"'
        )
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                fields = line.split('\t')
                if len(fields) >= 2:
                    taxid = fields[0]
                    lineage = fields[1]
                    # Remove cellular organisms prefix
                    if lineage.startswith('cellular organisms;'):
                        lineage = lineage.replace('cellular organisms;', '', 1)
                    taxid_to_lineage[taxid] = lineage
            
            logger.info(f"    ✓ Resolved {len(taxid_to_lineage)} taxids to lineages")
        else:
            logger.warning(f"    taxonkit failed")
        
        # Cleanup
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
    
    except Exception as e:
        logger.warning(f"    Error running taxonkit: {e}")
    
    # Extract species (last element in lineage)
    def extract_species(lineage):
        if pd.isna(lineage) or not lineage:
            return 'Unknown'
        ranks = lineage.split(';')
        # Species is typically the last rank
        if len(ranks) > 0:
            species = ranks[-1].strip()
            return species if species else 'Unknown'
        return 'Unknown'
    
    # Map taxids to species
    blast_df['lineage'] = blast_df['taxid'].map(taxid_to_lineage)
    blast_df['species'] = blast_df['lineage'].apply(extract_species)
    
    # Get top N species by number of unitigs (excluding Unknown)
    species_counts = blast_df[blast_df['species'] != 'Unknown']['species'].value_counts()
    top_species = species_counts.head(top_n).index.tolist()
    
    logger.info(f"  Top {top_n} species by unitig count:")
    for i, (species, count) in enumerate(species_counts.head(top_n).items(), 1):
        logger.info(f"    {i}. {species}: {count:,} unitigs")
    
    # Merge species info with loadings
    loading_with_species = loading_df.merge(
        blast_df[['unitig_id', 'species']],
        on='unitig_id',
        how='left'
    )
    loading_with_species['species'] = loading_with_species['species'].fillna('No BLAST hit')
    
    # Group into top N + 'Other'
    loading_with_species['color_category'] = loading_with_species['species'].apply(
        lambda x: x if x in top_species else ('No BLAST hit' if x == 'No BLAST hit' else 'Other species')
    )
    
    # Create plot
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    plot_categories = [cat for cat in top_species + ['Other species', 'No BLAST hit'] if cat != 'Unknown']
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(plot_categories)}
    
    # Plot each category
    for category in plot_categories:
        cat_data = loading_with_species[loading_with_species['color_category'] == category]
        
        if len(cat_data) == 0:
            continue
        
        # Different opacity for different categories
        if category == 'No BLAST hit':
            opacity = 0.1
            size = 2
        elif category == 'Other species':
            opacity = 0.2
            size = 2
        else:
            opacity = 0.4
            size = 3
        
        fig.add_trace(go.Scatter(
            x=cat_data['PC1_loading'],
            y=cat_data['PC2_loading'],
            mode='markers',
            marker=dict(
                size=size,
                color=category_colors[category],
                opacity=opacity,
                line=dict(width=0)
            ),
            name=f'{category} ({len(cat_data):,})',
            hovertemplate=f'<b>{category}</b><br>ID: %{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
            text=cat_data['unitig_id'],
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'Unitig PCA Loadings - Colored by Top {top_n} Species',
        xaxis_title=f'PC1 Loading ({var_pc1:.1f}% variance)',
        yaxis_title=f'PC2 Loading ({var_pc2:.1f}% variance)',
        template='plotly_white',
        width=1400,
        height=1000,
        font=dict(size=12),
        hovermode='closest',
        legend=dict(
            title='Species',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Save HTML
    html_path = output_dir / 'sup_04_pca_unitig_loadings_by_species.html'
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved: {html_path}")
    
    # Save PNG
    try:
        png_path = output_dir / 'sup_04_pca_unitig_loadings_by_species.png'
        fig.write_image(str(png_path), width=1400, height=1000, scale=2)
        logger.info(f"  ✓ Saved: {png_path}")
    except Exception as e:
        logger.warning(f"  Could not save PNG: {e}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("PCA Analysis of Unitig Matrix")
    logger.info("=" * 80)
    
    # Create output directories
    Path(PATHS['figures_dir']).mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    results_dir = Path(PATHS['feature_importance_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define cache paths
    pca_cache = Path('models/pca_reference.pkl')
    umap_cache = results_dir / 'umap_embedding_cache.pkl'
    tsne_cache = results_dir / 'tsne_embedding_cache.pkl'
    
    try:
        # Load data (standardization OFF by default - unitig fractions already normalized)
        matrix, sample_ids, unitig_ids, scaler = load_unitig_matrix(standardize=False)
        metadata, meta_indices = load_metadata(sample_ids)
        blast_annotations = load_blast_annotations(unitig_ids)
        
        # Perform PCA (load from cache if available)
        if pca_cache.exists():
            logger.info(f"\n✓ Loading cached PCA model: {pca_cache}")
            with open(pca_cache, 'rb') as f:
                cached = pickle.load(f)
            
            # Handle both old dict format and new raw model format
            if isinstance(cached, dict):
                pca = cached.get('pca_model', cached.get('pca'))
                if pca is None:
                    logger.warning("  Old cache format incompatible, recomputing...")
                    pca, pca_result = perform_pca(matrix, n_components=50)
                else:
                    # USE CACHED COORDINATES - DON'T RE-TRANSFORM!
                    pca_result = cached.get('pca_coordinates')
                    
                    if pca_result is None:
                        logger.warning("  No cached coordinates found, re-transforming...")
                        pca_result = pca.transform(matrix)
                    else:
                        logger.info(f"  ✓ Loaded cached PCA coordinates: {pca_result.shape}")
                        logger.info(f"  ✓ Loaded PCA model with {pca.n_components_} components (dict cache)")
            else:
                # Old format: just the model, no coordinates
                pca = cached
                logger.warning("  Old cache format (model only), re-transforming...")
                pca_result = pca.transform(matrix)
                logger.info(f"  Loaded PCA with {pca.n_components_} components")
        else:
            pca, pca_result = perform_pca(matrix, n_components=50)
        
        # Perform UMAP (load from cache if available)
        if umap_cache.exists():
            logger.info(f"\n✓ Loading cached UMAP embedding: {umap_cache}")
            with open(umap_cache, 'rb') as f:
                umap_data = pickle.load(f)
            # Handle both old format (coordinates only) and new format (model + coordinates)
            if isinstance(umap_data, dict):
                umap_model = umap_data['model']
                umap_result = umap_data['coordinates']
            else:
                # Old format - coordinates only
                umap_result = umap_data
                umap_model = None
            logger.info(f"  Loaded UMAP embedding: {umap_result.shape}")
        else:
            result = perform_umap(matrix, n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=N_JOBS)
            if result is not None:
                umap_model, umap_result = result
                # Save both model and coordinates for future projections
                with open(umap_cache, 'wb') as f:
                    pickle.dump({'model': umap_model, 'coordinates': umap_result}, f)
                logger.info(f"  ✓ Saved UMAP model and embedding: {umap_cache}")
            else:
                umap_model, umap_result = None, None
        
        # Perform t-SNE (load from cache if available)
        if tsne_cache.exists():
            logger.info(f"\n✓ Loading cached t-SNE embedding: {tsne_cache}")
            with open(tsne_cache, 'rb') as f:
                tsne_result = pickle.load(f)
            logger.info(f"  Loaded t-SNE embedding: {tsne_result.shape}")
        else:
            tsne_result = perform_tsne(matrix, n_components=2, perplexity=30)
            if tsne_result is not None:
                with open(tsne_cache, 'wb') as f:
                    pickle.dump(tsne_result, f)
                logger.info(f"  ✓ Saved t-SNE embedding: {tsne_cache}")
        
        # Align all embeddings to samples that have metadata (meta_indices may exclude
        # a small number of matrix samples not present in train/test metadata files).
        # Cached embeddings may already be aligned (len == len(meta_indices)); only
        # slice when the embedding covers the full unfiltered sample list.
        n_full = len(sample_ids)
        if pca_result.shape[0] == n_full:
            pca_result = pca_result[meta_indices]
        if umap_result is not None and umap_result.shape[0] == n_full:
            umap_result = umap_result[meta_indices]
        if tsne_result is not None and tsne_result.shape[0] == n_full:
            tsne_result = tsne_result[meta_indices]
        sample_ids = [sample_ids[i] for i in meta_indices]

        # Calculate separation metrics for PCA
        separation_metrics = calculate_separation_metrics(pca_result, metadata)
        
        # Calculate separation for UMAP if available
        if umap_result is not None:
            umap_metrics = calculate_separation_metrics(umap_result, metadata, n_pcs=2)
            separation_metrics['umap'] = umap_metrics
            # Export UMAP coordinates
            export_embedding_coordinates(umap_result, sample_ids, metadata, 'UMAP', results_dir)
        
        # Calculate separation for t-SNE if available
        if tsne_result is not None:
            tsne_metrics = calculate_separation_metrics(tsne_result, metadata, n_pcs=2)
            separation_metrics['tsne'] = tsne_metrics
            # Export t-SNE coordinates
            export_embedding_coordinates(tsne_result, sample_ids, metadata, 't-SNE', results_dir)
        
        # Calculate PC-task correlations
        pc_correlations = calculate_pc_task_correlations(pca_result, metadata, pca)
        
        # Save metrics
        metrics_path = results_dir / 'pca_separation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(separation_metrics, f, indent=2)
        logger.info(f"\n✓ Saved separation metrics: {metrics_path}")
        
        correlations_path = results_dir / 'pca_task_correlations.json'
        with open(correlations_path, 'w') as f:
            json.dump(pc_correlations, f, indent=2)
        logger.info(f"✓ Saved PC-task correlations: {correlations_path}")
        
        # Export PCA coordinates and loadings
        export_pca_coordinates(pca_result, sample_ids, metadata, pca, results_dir)
        export_pca_loadings(pca, unitig_ids, results_dir, top_n=100)
        
        # Save PCA model for future use
        save_pca_model(pca, pca_result, sample_ids, metadata, unitig_ids, scaler=scaler, 
                      umap_model=umap_model, umap_result=umap_result)
        
        # Create scree plot
        scree_path = Path(PATHS['figures_dir']) / "sup_03_pca_scree_plot.png"
        plot_scree_plot(pca, scree_path)
        
        # Create PCA loading plots (unitigs, not samples)
        # Load discriminant features from blast_annotations.tsv
        discriminant_file = Path('results/feature_analysis/blast_annotations.tsv')
        if discriminant_file.exists():
            feature_importance = pl.read_csv(discriminant_file, separator='\t')
            logger.info(f"\nLoaded discriminant features: {len(feature_importance)} rows")
            logger.info(f"  Columns: {feature_importance.columns}")
            if 'feature_index' in feature_importance.columns:
                logger.info(f"  ✓ feature_index column found for discriminant features")
            else:
                logger.warning("  ⚠ No feature_index column - discriminant plot will be skipped")
        else:
            logger.warning(f"  Discriminant features file not found: {discriminant_file}")
            feature_importance = None
        
        # Generate loading plots for all taxonomy levels
        taxonomy_levels = ['kingdom', 'phylum', 'class', 'order']
        for tax_level in taxonomy_levels:
            logger.info(f"\n{'='*60}")
            logger.info(f"Generating loading plots colored by {tax_level}...")
            logger.info(f"{'='*60}")
            plot_pca_loadings(
                pca, 
                unitig_ids, 
                blast_annotations, 
                Path(PATHS['figures_dir']),
                feature_importance=feature_importance,
                color_by=tax_level
            )
        
        # Create unitig PCA plot colored by top 10 species
        plot_unitig_pca_by_top_species(
            pca,
            unitig_ids,
            blast_annotations,
            Path(PATHS['figures_dir']),
            top_n=10
        )
        
        # Create PCA plots for each task (Supplementary Figure 3)
        for task in TASKS:
            if task in metadata.columns:
                plot_pca_by_task(pca_result, metadata, task, pca, 'sup_03_pca')
            else:
                logger.warning(f"  {task} not found in metadata")
        
        logger.info("\n" + "=" * 80)
        logger.info("PCA analysis complete!")
        logger.info("=" * 80)
        logger.info("\nGenerated files:")
        logger.info(f"  - PCA model: models/pca_reference.pkl")
        logger.info(f"  - Scree plot: {scree_path}")
        logger.info(f"  - Sample PCA plots (sup_03): {PATHS['figures_dir']}/sup_03_pca_*.png/html")
        logger.info(f"  - PCA loading plots (sup_04): {PATHS['figures_dir']}/sup_04_pca_loadings_*.png/html")
        logger.info(f"  - Separation metrics: {metrics_path}")
        logger.info(f"  - PC-task correlations: {correlations_path}")
        logger.info(f"  - PCA coordinates: {results_dir}/pca_coordinates.csv")
        logger.info(f"  - PCA loadings: {results_dir}/pca_loadings_*.csv")
        if umap_result is not None:
            logger.info(f"  - UMAP coordinates: {results_dir}/umap_coordinates.csv")
        if tsne_result is not None:
            logger.info(f"  - t-SNE coordinates: {results_dir}/tsne_coordinates.csv")
        logger.info(f"\nMatrix info:")
        logger.info(f"  - {len(sample_ids):,} samples (train + test)")
        logger.info(f"  - {len(unitig_ids):,} unitigs (features)")
        logger.info(f"  - Standardized: {scaler is not None}")
        logger.info(f"  - Unitig IDs properly excluded from PCA: \u2713")
        logger.info(f"  - Sample order verified against kmtricks.fof: \u2713")
        if blast_annotations is not None:
            logger.info(f"  - {len(blast_annotations):,} unitigs with BLAST hits")
        logger.info("\nUse 07_project_new_samples_pca.py to project new samples onto this reference PCA.")
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"FATAL ERROR: {e}")
        logger.error(f"{'='*80}")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate PCA/UMAP/t-SNE analysis of unitig matrix',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use cached results (default - fast, generates all 4 taxonomy level plots)
  python scripts/paper/06_generate_pca_analysis.py
  
  # Force recalculation of all embeddings
  python scripts/paper/06_generate_pca_analysis.py --no-cache
        """
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force recalculation of PCA/UMAP/t-SNE (ignore cached results)'
    )
    
    args = parser.parse_args()
    
    # Remove cache files if requested
    if args.no_cache:
        logger.info("--no-cache flag set: removing cached embeddings...")
        for cache_file in [
            'models/pca_reference.pkl',
            'results/feature_analysis/umap_embedding_cache.pkl',
            'results/feature_analysis/tsne_embedding_cache.pkl'
        ]:
            cache_path = Path(cache_file)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"  Removed: {cache_file}")
    
    main()
