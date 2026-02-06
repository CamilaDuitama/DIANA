#!/usr/bin/env python3
"""
Project new samples onto reference PCA space.

This script projects validation or new samples onto the PCA space computed from
training data, allowing comparison of new samples against the reference distribution.

CRITICAL FIXES:
- Uses pre-computed reference PCA coordinates (no re-loading of 1.6GB matrix)
- Applies same standardization as reference (saves/loads scaler)
- Validates unitig ID alignment (not just counts)
- Comprehensive error handling and validation
- Export projection results with similarity metrics

Usage:
    python 07_project_new_samples_pca.py --sample-matrix <path> --sample-ids <path> [--output <dir>]

Example:
    python 07_project_new_samples_pca.py \\
        --sample-matrix data/validation/validation_matrix.pa.mat \\
        --sample-ids data/validation/validation_sample_ids.txt \\
        --output paper/figures/pca_projection/
"""

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import pickle
import argparse
import sys
import os
import subprocess
import tempfile
import logging
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    tqdm = lambda x, **kwargs: x
from scipy.spatial.distance import cdist

# Import local config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'configs'))
from paper_config import PATHS, PLOT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_from_validation_dirs(validation_dirs):
    """Load samples from diana-predict output directories.
    
    Args:
        validation_dirs: List of paths to validation_predictions/<sample_id>/ directories
    
    Returns:
        matrix: (n_samples, n_features) array
        sample_ids: list of sample IDs
        unitig_ids: None (no unitig IDs in fraction files)
    """
    logger.info(f"\nLoading samples from validation prediction directories...")
    logger.info(f"  Number of directories: {len(validation_dirs)}")
    
    matrices = []
    sample_ids = []
    
    for vdir in validation_dirs:
        vdir_path = Path(vdir)
        if not vdir_path.exists():
            raise FileNotFoundError(f"Validation directory not found: {vdir}")
        
        # Extract sample ID from directory name
        sample_id = vdir_path.name
        
        # Look for unitig fraction file
        fraction_file = vdir_path / f"{sample_id}_unitig_fraction.txt"
        if not fraction_file.exists():
            raise FileNotFoundError(f"Unitig fraction file not found: {fraction_file}")
        
        # Load vector (column format)
        vector = np.loadtxt(fraction_file)
        matrices.append(vector)
        sample_ids.append(sample_id)
        
        logger.info(f"  ✓ Loaded {sample_id}: {vector.shape[0]} features")
    
    # Stack into matrix (samples × features)
    matrix = np.vstack(matrices)
    logger.info(f"\n  Final matrix shape: {matrix.shape} (samples × features)")
    
    return matrix, sample_ids, None


def load_pca_reference(pca_path):
    """
    Load saved PCA model and pre-computed reference coordinates.
    
    Returns:
        dict with keys: pca_model, pca_coordinates, scaler, sample_ids, 
                       unitig_ids, metadata, n_features, standardized
    """
    logger.info(f"Loading PCA reference from {pca_path}...")
    
    if not Path(pca_path).exists():
        raise FileNotFoundError(f"PCA reference file not found: {pca_path}")
    
    try:
        with open(pca_path, 'rb') as f:
            reference_data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load PCA reference: {e}")
    
    # Validate required keys
    required_keys = ['pca_model', 'sample_ids', 'unitig_ids', 'n_features']
    missing_keys = [k for k in required_keys if k not in reference_data]
    if missing_keys:
        raise ValueError(f"PCA reference missing required keys: {missing_keys}")
    
    logger.info(f"  ✓ PCA model with {reference_data['pca_model'].n_components_} components")
    logger.info(f"  ✓ Reference: {len(reference_data['sample_ids'])} samples")
    logger.info(f"  ✓ Features: {reference_data['n_features']} unitigs")
    
    # Check if we have pre-computed coordinates
    if 'pca_coordinates' in reference_data:
        logger.info(f"  ✓ Using pre-computed reference PCA coordinates")
    else:
        logger.warning(f"  ⚠ No pre-computed coordinates - will need reference matrix")
    
    # Check standardization status
    if reference_data.get('standardized', False):
        if 'scaler' in reference_data and reference_data['scaler'] is not None:
            logger.info(f"  ✓ Data was standardized (scaler available)")
        else:
            logger.warning(f"  ⚠ Data was standardized but scaler not saved!")
    else:
        logger.info(f"  ✓ Data was not standardized (raw values)")
    
    return reference_data


def load_new_samples(matrix_path, sample_ids_path, expected_format='samples_x_features'):
    """
    Load new sample matrix and IDs with validation.
    
    Args:
        matrix_path: Path to matrix file
        sample_ids_path: Path to sample IDs (one per line)
        expected_format: 'samples_x_features' or check first column for IDs
        
    Returns:
        matrix: (n_samples, n_features) array
        sample_ids: list of sample IDs
        unitig_ids: array of unitig IDs (if present in matrix)
    """
    logger.info(f"\nLoading new samples...")
    logger.info(f"  Matrix: {matrix_path}")
    logger.info(f"  Sample IDs: {sample_ids_path}")
    
    # Validate paths
    if not Path(matrix_path).exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")
    if not Path(sample_ids_path).exists():
        raise FileNotFoundError(f"Sample IDs file not found: {sample_ids_path}")
    
    # Load matrix
    try:
        data = np.loadtxt(matrix_path)
        logger.info(f"  Loaded matrix shape: {data.shape}")
        
        # Ensure 2D array (handle single sample case)
        if data.ndim == 1:
            logger.info(f"  ℹ Single sample detected, reshaping to 2D")
            data = data.reshape(1, -1)
            logger.info(f"  Reshaped to: {data.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load matrix: {e}")
    
    # Check if first column contains unitig IDs (like reference matrix)
    unitig_ids = None
    if data.shape[1] > data.shape[0]:  # More columns than rows suggests transposed
        logger.info(f"  Detecting matrix format...")
        # Check if first column looks like IDs (integers)
        first_col = data[:, 0]
        if np.all(first_col == first_col.astype(int)) and np.all(np.diff(first_col) > 0):
            logger.info(f"  ✓ First column detected as unitig IDs")
            unitig_ids = data[:, 0].astype(int)
            matrix = data[:, 1:].T  # Transpose to samples × unitigs
            logger.info(f"  ✓ Transposed to: {matrix.shape} (samples × unitigs)")
        else:
            matrix = data
    else:
        matrix = data
    
    # Load sample IDs
    try:
        with open(sample_ids_path) as f:
            sample_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to load sample IDs: {e}")
    
    # Validate alignment
    if len(sample_ids) != matrix.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {len(sample_ids)} IDs but matrix has {matrix.shape[0]} rows"
        )
    
    # Check for duplicates
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sid for sid in set(sample_ids) if sample_ids.count(sid) > 1]
        logger.warning(f"  ⚠ Duplicate sample IDs found: {duplicates}")
    
    # Check for NaN/inf
    nan_count = np.isnan(matrix).sum()
    inf_count = np.isinf(matrix).sum()
    if nan_count > 0:
        logger.warning(f"  ⚠ Found {nan_count} NaN values - replacing with 0")
        matrix = np.nan_to_num(matrix, nan=0.0)
    if inf_count > 0:
        logger.warning(f"  ⚠ Found {inf_count} infinite values - replacing with 0")
        matrix = np.nan_to_num(matrix, posinf=0.0, neginf=0.0)
    
    logger.info(f"  ✓ Loaded {len(sample_ids)} new samples")
    logger.info(f"  ✓ Matrix shape: {matrix.shape} (samples × features)")
    
    return matrix, sample_ids, unitig_ids


def validate_feature_alignment(new_matrix, new_unitig_ids, ref_unitig_ids):
    """
    Validate that new samples have the same unitigs as reference in same order.
    
    Returns:
        aligned_matrix: Matrix reordered/filtered to match reference
        alignment_info: Dict with alignment statistics
    """
    logger.info("\nValidating feature alignment...")
    
    alignment_info = {
        'n_features_new': new_matrix.shape[1],
        'n_features_ref': len(ref_unitig_ids),
        'perfect_match': False,
        'reordered': False,
        'missing_features': 0,
        'extra_features': 0
    }
    
    # If no unitig IDs in new matrix, assume same order as reference
    if new_unitig_ids is None:
        logger.warning(
            "⚠️  NO UNITIG IDs! Assuming SAME ORDER as reference. "
            "If order differs, results will be WRONG!"
        )
        if new_matrix.shape[1] != len(ref_unitig_ids):
            raise ValueError(
                f"Feature count mismatch: new matrix has {new_matrix.shape[1]} features "
                f"but reference has {len(ref_unitig_ids)} features. "
                f"Cannot verify alignment without unitig IDs."
            )
        alignment_info['perfect_match'] = True
        return new_matrix, alignment_info
    
    # Convert to sets for comparison
    new_ids_set = set(new_unitig_ids)
    ref_ids_set = set(ref_unitig_ids)
    
    # Check for perfect match
    if list(new_unitig_ids) == list(ref_unitig_ids):
        logger.info(f"  ✓ Perfect alignment: all {len(ref_unitig_ids)} unitigs in correct order")
        alignment_info['perfect_match'] = True
        return new_matrix, alignment_info
    
    # Check if same unitigs but different order
    if new_ids_set == ref_ids_set:
        logger.info(f"  Reordering features to match reference...")
        # Create mapping from ref unitig ID to new matrix column index
        new_id_to_idx = {uid: idx for idx, uid in enumerate(new_unitig_ids)}
        new_order = [new_id_to_idx[uid] for uid in ref_unitig_ids]
        aligned_matrix = new_matrix[:, new_order]
        alignment_info['reordered'] = True
        logger.info(f"  ✓ Reordered {len(ref_unitig_ids)} features to match reference")
        return aligned_matrix, alignment_info
    
    # Handle missing/extra features
    missing = ref_ids_set - new_ids_set
    extra = new_ids_set - ref_ids_set
    
    alignment_info['missing_features'] = len(missing)
    alignment_info['extra_features'] = len(extra)
    
    if missing:
        logger.warning(f"  ⚠ {len(missing)} unitigs missing from new samples (will fill with zeros)")
    if extra:
        logger.warning(f"  ⚠ {len(extra)} extra unitigs in new samples (will be ignored)")
    
    # Create aligned matrix
    aligned_matrix = np.zeros((new_matrix.shape[0], len(ref_unitig_ids)))
    new_id_to_idx = {uid: idx for idx, uid in enumerate(new_unitig_ids)}
    
    for ref_idx, ref_uid in enumerate(ref_unitig_ids):
        if ref_uid in new_id_to_idx:
            new_idx = new_id_to_idx[ref_uid]
            aligned_matrix[:, ref_idx] = new_matrix[:, new_idx]
        # else: stays zero (missing feature)
    
    logger.info(f"  ✓ Aligned matrix: {aligned_matrix.shape}")
    return aligned_matrix, alignment_info


def project_samples(pca_model, new_matrix, scaler=None):
    """
    Project new samples onto PCA space with proper standardization.
    
    Args:
        pca_model: Fitted PCA model
        new_matrix: (n_samples, n_features) array
        scaler: StandardScaler or None
        
    Returns:
        pca_projected: (n_samples, n_components) PCA coordinates
    """
    logger.info("\nProjecting new samples onto PCA space...")
    
    # Validate feature dimensions
    if hasattr(pca_model, 'n_features_in_'):
        if new_matrix.shape[1] != pca_model.n_features_in_:
            raise ValueError(
                f"PCA expects {pca_model.n_features_in_} features, "
                f"got {new_matrix.shape[1]}"
            )
    
    # Apply same standardization as reference
    if scaler is not None:
        # Validate scaler dimensions
        if hasattr(scaler, 'n_features_in_'):
            if new_matrix.shape[1] != scaler.n_features_in_:
                raise ValueError(
                    f"Scaler expects {scaler.n_features_in_} features, "
                    f"got {new_matrix.shape[1]}"
                )
        logger.info("  Applying reference standardization...")
        new_matrix_std = scaler.transform(new_matrix)
    else:
        logger.info("  No standardization (using raw values)")
        new_matrix_std = new_matrix
    
    # Project onto PCA space
    try:
        pca_projected = pca_model.transform(new_matrix_std)
    except Exception as e:
        raise RuntimeError(f"PCA projection failed: {e}")
    
    logger.info(f"  ✓ Projected shape: {pca_projected.shape}")
    logger.info(f"  ✓ PC ranges - PC1: [{pca_projected[:, 0].min():.2f}, {pca_projected[:, 0].max():.2f}], "
                f"PC2: [{pca_projected[:, 1].min():.2f}, {pca_projected[:, 1].max():.2f}]")
    
    return pca_projected


def compute_similarity_metrics(new_pca, ref_pca, ref_sample_ids, k=5):
    """
    Compute similarity of new samples to reference samples.
    
    Returns:
        DataFrame with columns: sample_id, nearest_neighbors, distances, is_outlier
    """
    logger.info(f"\nComputing similarity metrics (k={k} nearest neighbors)...")
    
    # Compute pairwise distances (optimized for large batches)
    if new_pca.shape[0] > 100:
        logger.info(f"  Processing {new_pca.shape[0]} samples in batches...")
        distances = []
        for i in tqdm(range(new_pca.shape[0]), desc="Computing distances"):
            dist = cdist([new_pca[i]], ref_pca, metric='euclidean')[0]
            distances.append(dist)
        distances = np.array(distances)
    else:
        distances = cdist(new_pca, ref_pca, metric='euclidean')
    
    # Find k nearest neighbors for each new sample
    results = []
    for i in range(new_pca.shape[0]):
        sample_distances = distances[i, :]
        nearest_indices = np.argsort(sample_distances)[:k]
        nearest_distances = sample_distances[nearest_indices]
        nearest_ids = [ref_sample_ids[idx] for idx in nearest_indices]
        
        # Simple outlier detection: distance to nearest > 2 std of all distances
        mean_dist = distances.mean()
        std_dist = distances.std()
        is_outlier = nearest_distances[0] > (mean_dist + 2 * std_dist)
        
        results.append({
            'nearest_sample_ids': nearest_ids,
            'nearest_distances': nearest_distances.tolist(),
            'mean_distance_to_k_nearest': nearest_distances.mean(),
            'is_outlier': is_outlier
        })
    
    logger.info(f"  ✓ Computed distances to {len(ref_sample_ids)} reference samples")
    outliers = sum(r['is_outlier'] for r in results)
    if outliers > 0:
        logger.warning(f"  ⚠ {outliers} samples flagged as potential outliers")
    
    return results


def export_projection_results(new_pca, new_sample_ids, similarity_metrics, output_dir, ref_pca=None, ref_sample_ids=None, export_reference=False):
    """Export projection coordinates and similarity metrics to CSV."""
    logger.info("\nExporting projection results...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export new sample coordinates
    coords_df = pd.DataFrame(
        new_pca,
        columns=[f'PC{i+1}' for i in range(new_pca.shape[1])]
    )
    coords_df.insert(0, 'sample_id', new_sample_ids)
    
    # Add similarity metrics
    for i, metrics in enumerate(similarity_metrics):
        coords_df.loc[i, 'nearest_neighbors'] = ','.join(metrics['nearest_sample_ids'][:3])
        coords_df.loc[i, 'mean_distance_to_neighbors'] = metrics['mean_distance_to_k_nearest']
        coords_df.loc[i, 'is_outlier'] = metrics['is_outlier']
    
    coords_path = output_dir / 'new_sample_pca_coordinates.csv'
    coords_df.to_csv(coords_path, index=False)
    logger.info(f"  ✓ Saved coordinates: {coords_path}")
    
    # Export combined coordinates only if requested (optional for large reference sets)
    if export_reference and ref_pca is not None and ref_sample_ids is not None:
        logger.info("  Exporting combined coordinates (including reference)...")
        ref_coords_df = pd.DataFrame(
            ref_pca,
            columns=[f'PC{i+1}' for i in range(ref_pca.shape[1])]
        )
        ref_coords_df.insert(0, 'sample_id', ref_sample_ids)
        ref_coords_df['dataset'] = 'reference'
        
        coords_df['dataset'] = 'new'
        combined_df = pd.concat([ref_coords_df, coords_df], ignore_index=True)
        
        combined_path = output_dir / 'combined_pca_coordinates.csv'
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"  ✓ Saved combined coordinates: {combined_path}")


def resolve_taxids(taxids, n_threads=None):
    """Resolve taxids to full lineages using taxonkit.
    
    Args:
        taxids: Array-like of taxid strings
        n_threads: Number of threads (default: from OMP_NUM_THREADS env var)
    
    Returns:
        dict mapping taxid (str) -> lineage (str)
    """
    taxid_to_lineage = {}
    temp_input = None
    
    try:
        # Create temp file with taxids
        temp_input = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        for taxid in taxids:
            temp_input.write(f"{taxid}\n")
        temp_input.close()
        
        # Get number of threads
        if n_threads is None:
            n_threads = os.environ.get('OMP_NUM_THREADS', '16')
        
        # Run taxonkit
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
                    # Remove 'cellular organisms;' prefix
                    if lineage.startswith('cellular organisms;'):
                        lineage = lineage.replace('cellular organisms;', '', 1)
                    taxid_to_lineage[taxid] = lineage
        
        # Cleanup
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
            
    except Exception as e:
        logger.warning(f"  Error running taxonkit: {e}")
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
    
    return taxid_to_lineage


def load_validation_metadata(sample_ids):
    """Load metadata for validation samples if available."""
    logger.info("\nLoading validation metadata...")
    
    val_meta_path = Path(PATHS['validation_metadata'])
    if not val_meta_path.exists():
        logger.warning("  Validation metadata not found - plots will use generic labels")
        return None
    
    try:
        metadata = pl.read_csv(val_meta_path, separator="\t")
        metadata = metadata.filter(pl.col("Run_accession").is_in(sample_ids))
        logger.info(f"  ✓ Loaded metadata for {len(metadata)} samples")
        return metadata
    except Exception as e:
        logger.error(f"  Failed to load metadata: {e}")
        return None


def plot_feature_evidence(sample_id, sample_unitig_fraction, discriminant_unitig_ids, ref_unitig_ids,
                          blast_annotations, output_dir, threshold=0.0001):
    """Plot discriminant marker presence/absence by genus for a sample.
    
    Args:
        sample_id: Sample identifier
        sample_unitig_fraction: Array of unitig fractions for this sample
        discriminant_unitig_ids: List of discriminant marker unitig IDs (top 400)
        ref_unitig_ids: All unitig IDs in order
        blast_annotations: BLAST DataFrame with unitig_id and taxid
        output_dir: Output directory
        threshold: Minimum fraction to consider unitig "present"
    """
    logger.info(f"\\nCreating Feature Evidence plot for {sample_id}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify present unitigs in sample
    present_mask = sample_unitig_fraction > threshold
    present_unitig_ids = ref_unitig_ids[present_mask]
    
    logger.info(f"  Sample has {len(present_unitig_ids)} present unitigs (>{threshold})")
    if len(present_unitig_ids) > 0:
        logger.info(f"  Present unitig ID range: {present_unitig_ids.min()} to {present_unitig_ids.max()}")
    logger.info(f"  Total discriminant markers: {len(discriminant_unitig_ids)}")
    if len(discriminant_unitig_ids) > 0:
        logger.info(f"  Discriminant unitig ID range: {min(discriminant_unitig_ids)} to {max(discriminant_unitig_ids)}")
    
    # Find which discriminant markers are present
    discriminant_set = set(discriminant_unitig_ids)
    present_set = set(present_unitig_ids)
    present_discriminant = discriminant_set & present_set
    absent_discriminant = discriminant_set - present_set
    
    logger.info(f"  Present discriminant markers: {len(present_discriminant)}")
    logger.info(f"  Absent discriminant markers: {len(absent_discriminant)}")
    
    # Get phylum for each discriminant marker
    blast_df = blast_annotations.copy()
    blast_df['taxid'] = blast_df['taxid'].astype(str)
    
    logger.info(f"  BLAST file has {len(blast_df)} annotations")
    blast_unitig_ids = blast_df['unitig_id'].unique()
    logger.info(f"  BLAST unitig ID range: {blast_unitig_ids.min()} to {blast_unitig_ids.max()}")
    
    # Filter to discriminant markers only
    discriminant_blast = blast_df[blast_df['unitig_id'].isin(discriminant_unitig_ids)].copy()
    logger.info(f"  Found {len(discriminant_blast)} BLAST hits for {len(discriminant_unitig_ids)} discriminant markers")
    
    # Resolve taxids to lineages using taxonkit
    unique_taxids = discriminant_blast['taxid'].dropna().unique()
    logger.info(f"  Resolving {len(unique_taxids)} taxids for discriminant markers...")
    
    taxid_to_lineage = resolve_taxids(unique_taxids)
    
    # Extract genus (second to last element in lineage)
    discriminant_blast['lineage'] = discriminant_blast['taxid'].map(taxid_to_lineage)
    discriminant_blast['genus'] = discriminant_blast['lineage'].apply(
        lambda x: x.split(';')[-2].strip() if pd.notna(x) and len(x.split(';')) > 1 else 'Unknown'
    )
    
    # Calculate top 10 genera IN THE SAMPLE (from all present unitigs, not just discriminant)
    # Get all present unitigs in sample
    present_unitig_df = pd.DataFrame({
        'unitig_id': present_unitig_ids,
        'fraction': sample_unitig_fraction[present_mask]
    })
    
    # Merge with ALL BLAST data to get genus for all present unitigs
    all_blast_with_genus = blast_df[['unitig_id', 'taxid']].drop_duplicates('unitig_id').copy()
    all_blast_with_genus['lineage'] = all_blast_with_genus['taxid'].map(taxid_to_lineage)
    all_blast_with_genus['genus'] = all_blast_with_genus['lineage'].apply(
        lambda x: x.split(';')[-2].strip() if pd.notna(x) and len(x.split(';')) > 1 else 'Unknown'
    )
    
    # Merge with present unitig fractions
    present_with_genus = present_unitig_df.merge(all_blast_with_genus[['unitig_id', 'genus']], on='unitig_id', how='left')
    
    # Calculate genus abundance in sample
    genus_abundance = present_with_genus.groupby('genus')['fraction'].sum().sort_values(ascending=False)
    
    # Filter out Unknown/unclassified and get top 10
    genus_abundance = genus_abundance[~genus_abundance.index.str.contains('Unknown|unclassified', case=False, na=False)]
    top_10_genera = genus_abundance.head(10).index.tolist()
    
    logger.info(f"  Top 10 genera in sample (by abundance): {top_10_genera}")
    
    # Count present/absent DISCRIMINANT markers by genus (only for top 10 genera in sample)
    genus_counts = []
    for genus in top_10_genera:
        genus_unitigs = discriminant_blast[discriminant_blast['genus'] == genus]['unitig_id'].unique()
        present_count = len(set(genus_unitigs) & present_discriminant)
        absent_count = len(set(genus_unitigs) & absent_discriminant)
        if present_count > 0 or absent_count > 0:  # Only include if there are discriminant markers
            genus_counts.append({
                'genus': genus,
                'present': present_count,
                'absent': absent_count,
                'total': present_count + absent_count
            })
    
    # Handle case where no genus found
    if not genus_counts:
        logger.warning(f"  No discriminant markers found for top 10 genera in sample")
        logger.warning(f"  Skipping Feature Evidence plot for {sample_id}")
        return
    
    genus_df = pd.DataFrame(genus_counts).sort_values('total', ascending=True)
    
    logger.info(f"  Plotting {len(genus_df)} top genera with discriminant markers")
    
    # Create stacked horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=genus_df['genus'],
        x=genus_df['present'],
        name='Present',
        orientation='h',
        marker=dict(color='steelblue'),
        text=genus_df['present'],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        y=genus_df['genus'],
        x=genus_df['absent'],
        name='Absent',
        orientation='h',
        marker=dict(color='lightgray'),
        text=genus_df['absent'],
        textposition='inside'
    ))
    
    fig.update_layout(
        title=f'Discriminant Marker Evidence - {sample_id}',
        xaxis_title='Number of Markers',
        yaxis_title='Genus',
        barmode='stack',
        template='plotly_white',
        width=1000,
        height=max(400, len(genus_df) * 30),
        showlegend=True,
        legend=dict(x=0.7, y=0.98)
    )
    
    html_path = output_dir / f"{sample_id}_feature_evidence.html"
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved: {html_path}")
    
    try:
        png_path = output_dir / f"{sample_id}_feature_evidence.png"
        fig.write_image(str(png_path), width=1000, height=max(400, len(genus_df) * 30), scale=2)
        logger.info(f"  ✓ Saved: {png_path}")
    except Exception as e:
        logger.warning(f"  ⚠ Could not save PNG: {e}")


def plot_taxonomy_profile(sample_id, sample_unitig_fraction, ref_unitig_ids,
                          blast_annotations, output_dir, threshold=0.0001, top_n=10):
    """Plot top N most abundant species in a sample.
    
    Args:
        sample_id: Sample identifier
        sample_unitig_fraction: Array of unitig fractions for this sample
        ref_unitig_ids: All unitig IDs in order
        blast_annotations: BLAST DataFrame with unitig_id and taxid
        output_dir: Output directory
        threshold: Minimum fraction to include unitig
        top_n: Number of top species to show
    """
    logger.info(f"\\nCreating Taxonomy Profile for {sample_id}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter unitigs above threshold
    above_threshold = sample_unitig_fraction > threshold
    filtered_fractions = sample_unitig_fraction[above_threshold]
    filtered_unitig_ids = ref_unitig_ids[above_threshold]
    
    logger.info(f"  {len(filtered_unitig_ids)} unitigs above threshold {threshold}")
    
    # Get BLAST annotations
    blast_df = blast_annotations.copy()
    blast_df['taxid'] = blast_df['taxid'].astype(str)
    
    # Merge with fractions
    unitig_abundance = pd.DataFrame({
        'unitig_id': filtered_unitig_ids,
        'fraction': filtered_fractions
    })
    
    unitig_with_tax = unitig_abundance.merge(
        blast_df[['unitig_id', 'taxid']].drop_duplicates('unitig_id'),
        on='unitig_id',
        how='left'
    )
    
    # Resolve taxids to species
    unique_taxids = unitig_with_tax['taxid'].dropna().unique()
    logger.info(f"  Resolving {len(unique_taxids)} taxids to species...")
    
    taxid_to_lineage = resolve_taxids(unique_taxids)
    logger.info(f"  Taxonkit resolved {len(taxid_to_lineage)} out of {len(unique_taxids)} taxids")
    
    if len(taxid_to_lineage) == 0:
        logger.warning(f"  Taxonkit returned 0 lineages - check if taxonkit module is loaded")
        logger.warning(f"  Skipping Taxonomy Profile plot for {sample_id}")
        return
    
    # Extract species (last element)
    unitig_with_tax['lineage'] = unitig_with_tax['taxid'].map(taxid_to_lineage)
    unitig_with_tax['species'] = unitig_with_tax['lineage'].apply(
        lambda x: x.split(';')[-1].strip() if pd.notna(x) and x else 'Unknown'
    )
    
    # Count unitigs by species (not sum of fractions)
    species_unitig_counts = unitig_with_tax.groupby('species')['unitig_id'].count().reset_index()
    species_unitig_counts.columns = ['species', 'unitig_count']
    species_unitig_counts = species_unitig_counts.sort_values('unitig_count', ascending=False)
    
    # Calculate total number of present unitigs for normalization
    total_unitigs = len(filtered_unitig_ids)
    
    # Log species distribution
    n_unknown_unitigs = species_unitig_counts[species_unitig_counts['species'] == 'Unknown']['unitig_count'].sum()
    n_known_unitigs = species_unitig_counts[species_unitig_counts['species'] != 'Unknown']['unitig_count'].sum()
    logger.info(f"  Species breakdown: {len(species_unitig_counts[species_unitig_counts['species'] != 'Unknown'])} identified, "
                f"{len(species_unitig_counts[species_unitig_counts['species'] == 'Unknown'])} unknown")
    logger.info(f"  Unitig breakdown: {n_known_unitigs/total_unitigs*100:.1f}% identified, {n_unknown_unitigs/total_unitigs*100:.1f}% unknown")
    
    # Filter out Unknown and take top N
    species_unitig_counts = species_unitig_counts[species_unitig_counts['species'] != 'Unknown']
    top_species = species_unitig_counts.head(top_n)
    
    # Handle case where no species found
    if len(top_species) == 0:
        logger.warning(f"  No identified species found (all Unknown or no BLAST hits)")
        logger.warning(f"  Skipping Taxonomy Profile plot for {sample_id}")
        return
    
    # Convert to relative abundance (percentage of total unitigs)
    top_species['percentage'] = (top_species['unitig_count'] / total_unitigs * 100).round(2)
    
    top_total_unitigs = top_species['unitig_count'].sum()
    logger.info(f"  Top {top_n} species represent {top_total_unitigs/total_unitigs*100:.1f}% of unitigs")
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_species['species'][::-1],  # Reverse for top-to-bottom
        x=top_species['percentage'][::-1],
        orientation='h',
        marker=dict(color='teal'),
        text=top_species['percentage'][::-1].apply(lambda x: f'{x:.1f}%'),
        textposition='inside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Species - {sample_id}',
        xaxis_title='Relative Abundance (%)',
        yaxis_title='Species',
        template='plotly_white',
        width=1000,
        height=max(400, top_n * 40),
        showlegend=False
    )
    
    html_path = output_dir / f"{sample_id}_taxonomy_profile.html"
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved: {html_path}")
    
    try:
        png_path = output_dir / f"{sample_id}_taxonomy_profile.png"
        fig.write_image(str(png_path), width=1000, height=max(400, top_n * 40), scale=2)
        logger.info(f"  ✓ Saved: {png_path}")
    except Exception as e:
        logger.warning(f"  ⚠ Could not save PNG: {e}")


def plot_unitig_pca_with_taxonomy(sample_id, sample_unitig_fraction, ref_unitig_pca, 
                                   ref_unitig_ids, blast_annotations, taxonomy_level='phylum',
                                   output_dir=None, top_n=10):
    """
    Plot unitig PCA loadings colored by top N taxa at specified taxonomy level.
    Shows where the new sample's unitigs land in the reference unitig PC space.
    
    NOTE: This function is NOT called by main() by default.
    To use: Enable with --plot-unitig-taxonomy flag (requires BLAST annotations)
    Or import directly: from this_module import plot_unitig_pca_with_taxonomy
    
    Args:
        sample_id: Sample identifier (e.g., 'ERR10114862')
        sample_unitig_fraction: Array of unitig fractions for this sample (length = n_unitigs)
        ref_unitig_pca: Reference unitig PCA model (from 06_generate_pca_analysis.py)
        ref_unitig_ids: Array of unitig IDs corresponding to PC loadings
        blast_annotations: DataFrame with columns ['unitig_id', 'taxid', ...] 
        taxonomy_level: 'kingdom', 'phylum', 'class', or 'order'
        output_dir: Where to save plots
        top_n: Number of top taxa to color (default 10)
    
    Returns:
        dict with plot paths and taxonomy stats
    """
    import subprocess
    import tempfile
    
    logger.info(f"\\nCreating unitig PCA plot for {sample_id} (colored by {taxonomy_level})...")
    
    if output_dir is None:
        output_dir = Path('paper/figures/unitig_pca')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get PC1 and PC2 loadings
    # ref_unitig_pca is now an array of shape (n_unitigs, n_components)
    # So PC1 loadings are column 0, PC2 loadings are column 1
    pc1_loadings = ref_unitig_pca[:, 0]
    pc2_loadings = ref_unitig_pca[:, 1]
    
    # We don't have explained variance from this array, so skip it in labels
    # (or you can pass the pca_model separately if needed)
    
    # Create unitig loading DataFrame
    loading_df = pd.DataFrame({
        'unitig_id': ref_unitig_ids,
        'PC1_loading': pc1_loadings,
        'PC2_loading': pc2_loadings
    })
    
    # Convert blast_annotations to pandas if needed
    if hasattr(blast_annotations, 'to_pandas'):
        blast_df = blast_annotations.to_pandas()
    else:
        blast_df = blast_annotations
    
    # Ensure correct dtypes
    loading_df['unitig_id'] = loading_df['unitig_id'].astype('int64')
    
    logger.info(f"  BLAST dataframe shape: {blast_df.shape}")
    logger.info(f"  Loading dataframe shape: {loading_df.shape}")
    
    # Get taxonomy from taxids
    logger.info(f"  Resolving taxonomy for {len(blast_df)} BLAST hits...")
    
    if 'taxid' not in blast_df.columns:
        logger.warning("  No taxid column - cannot create taxonomy plot")
        return None
    
    # Convert taxid to string for taxonkit
    blast_df['taxid'] = blast_df['taxid'].astype(str)
    
    # Get unique taxids
    unique_taxids = blast_df['taxid'].dropna().unique()
    logger.info(f"  Found {len(unique_taxids)} unique taxids")
    
    # Resolve taxids using helper function
    logger.info(f"  Running taxonkit...")
    taxid_to_lineage = resolve_taxids(unique_taxids)
    logger.info(f"  ✓ Resolved {len(taxid_to_lineage)} taxids to lineages")
    
    # Extract taxonomy level from lineages
    def extract_taxonomy_level(lineage, level):
        """Extract specific taxonomy level from semicolon-separated lineage."""
        if pd.isna(lineage) or not lineage:
            return 'Unknown'
        
        ranks = lineage.split(';')
        
        # For species, just take the last element
        if level == 'species':
            return ranks[-1].strip() if ranks else 'Unknown'
        
        # For other levels, try to parse position
        # Typical order: domain/kingdom; phylum; class; order; family; genus; species
        level_map = {'kingdom': 0, 'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        idx = level_map.get(level, -1)
        
        if idx >= 0 and idx < len(ranks):
            return ranks[idx].strip() if ranks[idx].strip() else 'Unknown'
        return 'Unknown'
    
    # Add lineage to blast_df
    blast_df['lineage'] = blast_df['taxid'].map(taxid_to_lineage)
    blast_df['taxonomy_category'] = blast_df['lineage'].apply(
        lambda x: extract_taxonomy_level(x, taxonomy_level)
    )
    
    # Identify sample's present unitigs (where fraction > 0)
    sample_unitig_mask = sample_unitig_fraction > 0
    sample_unitig_ids = ref_unitig_ids[sample_unitig_mask]
    sample_fractions = sample_unitig_fraction[sample_unitig_mask]
    
    logger.info(f"  Sample {sample_id} has {len(sample_unitig_ids)} present unitigs")
    
    # Filter BLAST to only sample's present unitigs
    sample_blast = blast_df[blast_df['unitig_id'].isin(sample_unitig_ids)].copy()
    
    if len(sample_blast) == 0:
        logger.warning(f"  No BLAST hits for sample's present unitigs")
        return None
    
    # Get top N taxa IN THIS SAMPLE by abundance (excluding Unknown/unclassified)
    sample_with_fractions = pd.DataFrame({
        'unitig_id': sample_unitig_ids,
        'fraction': sample_fractions
    })
    sample_blast_with_frac = sample_blast.merge(sample_with_fractions, on='unitig_id', how='left')
    
    # Group by taxonomy and sum fractions
    taxa_abundance = sample_blast_with_frac.groupby('taxonomy_category')['fraction'].sum().sort_values(ascending=False)
    
    # Filter out Unknown/unclassified
    taxa_abundance = taxa_abundance[~taxa_abundance.index.str.contains('Unknown|unclassified', case=False, na=False)]
    top_taxa = taxa_abundance.head(top_n).index.tolist()
    
    logger.info(f"  Top {top_n} {taxonomy_level}s in sample (by abundance): {top_taxa}")
    
    # Filter sample_blast to ONLY unitigs belonging to top N species
    top_taxa_blast = sample_blast[sample_blast['taxonomy_category'].isin(top_taxa)].copy()
    
    # Keep only BEST hit per unitig (highest bitscore)
    top_taxa_blast = top_taxa_blast.sort_values('bitscore', ascending=False).drop_duplicates('unitig_id', keep='first')
    top_taxa_unitig_ids = top_taxa_blast['unitig_id'].unique()
    
    logger.info(f"  Found {len(top_taxa_unitig_ids)} unitigs from top {top_n} {taxonomy_level}s")
    
    # Merge taxonomy with loadings (only for top taxa unitigs, already deduplicated)
    loading_with_tax = loading_df[loading_df['unitig_id'].isin(top_taxa_unitig_ids)].copy()
    loading_with_tax = loading_with_tax.merge(
        top_taxa_blast[['unitig_id', 'taxonomy_category']], 
        on='unitig_id', 
        how='left'
    )
    loading_with_tax['taxonomy_category'] = loading_with_tax['taxonomy_category'].fillna('Other/Unknown')
    
    # Use taxonomy_category directly for coloring (no grouping needed since we already filtered)
    loading_with_tax['color_category'] = loading_with_tax['taxonomy_category']
    
    # Final filter to remove any Unknown that slipped through
    sample_data = loading_with_tax[~loading_with_tax['color_category'].str.contains('Unknown|unclassified', case=False, na=False)].copy()
    
    if len(sample_data) == 0:
        logger.warning(f"  No present unitigs found for {sample_id}")
        return None
    
    logger.info(f"  Plotting {len(sample_data)} unitigs (from top {top_n} {taxonomy_level}s)")
    
    # Create plot
    fig = go.Figure()
    
    # Color palette
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(top_taxa + ['Other/Unknown'])}
    
    # Plot sample's unitigs by taxonomy category (excluding Unknown)
    plot_categories = [cat for cat in top_taxa + ['Other/Unknown'] if cat != 'Unknown']
    for category in plot_categories:
        cat_data = sample_data[sample_data['color_category'] == category]
        
        if len(cat_data) == 0:
            continue
        
        fig.add_trace(go.Scattergl(
            x=cat_data['PC1_loading'],
            y=cat_data['PC2_loading'],
            mode='markers',
            marker=dict(
                size=5,
                color=category_colors[category],
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            name=f'{category} ({len(cat_data):,})',
            hovertemplate=f'<b>{category}</b><br>ID: %{{text}}<br>PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>',
            text=cat_data['unitig_id'],
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'Unitig PCA Loadings - {sample_id} (colored by {taxonomy_level.capitalize()})',
        xaxis_title='PC1 Loading',
        yaxis_title='PC2 Loading',
        template='plotly_white',
        width=1400,
        height=1000,
        font=dict(size=12),
        hovermode='closest',
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    # Save HTML
    html_path = output_dir / f'{sample_id}_unitig_pca_{taxonomy_level}.html'
    fig.write_html(str(html_path))
    logger.info(f"  ✓ Saved: {html_path}")
    
    # Save PNG
    try:
        png_path = output_dir / f'{sample_id}_unitig_pca_{taxonomy_level}.png'
        fig.write_image(str(png_path), width=1400, height=1000, scale=2)
        logger.info(f"  ✓ Saved: {png_path}")
    except Exception as e:
        logger.warning(f"  Could not save PNG: {e}")
        png_path = None
    
    return {
        'html_path': html_path,
        'png_path': png_path,
        'top_taxa': top_taxa,
        'n_sample_unitigs': len(sample_unitig_ids),
        'taxonomy_level': taxonomy_level
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Project new samples onto reference PCA space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Project from matrix files
  python 07_project_new_samples_pca.py \\
      --sample-matrix data/validation/validation_matrix.pa.mat \\
      --sample-ids data/validation/validation_sample_ids.txt \\
      --output paper/figures/pca_projection/
  
  # Project from diana-predict output (single sample)
  python 07_project_new_samples_pca.py \\
      --validation-dir results/validation_predictions/ERR3609654 \\
      --output paper/figures/pca_projection_test/
  
  # Project from diana-predict output (multiple samples)
  python 07_project_new_samples_pca.py \\
      --validation-dir results/validation_predictions/ERR3609654 results/validation_predictions/ERR3678185 \\
      --output paper/figures/pca_projection_test/
  
  # Use custom BLAST and discriminant features paths
  python 07_project_new_samples_pca.py \\
      --validation-dir results/validation_predictions/ERR3609654 \\
      --blast-results path/to/blast_results.txt \\
      --discriminant-features path/to/features.tsv \\
      --output paper/figures/pca_projection_test/
        """
    )
    
    # Input group: either matrix files OR validation directories
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sample-matrix', help='Path to sample matrix file')
    input_group.add_argument('--validation-dir', nargs='+', 
                           help='Path(s) to diana-predict output directory/directories (e.g., results/validation_predictions/SAMPLE_ID)')
    
    parser.add_argument('--sample-ids', help='Path to sample IDs file (required with --sample-matrix)')
    parser.add_argument('--pca-model', default='models/pca_reference.pkl', 
                        help='Path to saved PCA model (default: models/pca_reference.pkl)')
    parser.add_argument('--output', default=None, 
                        help='Output directory (default: paper/figures/pca_projection/)')
    parser.add_argument('--k-neighbors', type=int, default=5,
                        help='Number of nearest neighbors for similarity (default: 5)')
    parser.add_argument('--export-reference', action='store_true',
                        help='Export combined CSV with reference samples (large file)')
    parser.add_argument('--blast-results', 
                        default='results/feature_analysis/all_features_blast/blast_results.txt',
                        help='Path to BLAST results TSV (default: results/feature_analysis/all_features_blast/blast_results.txt)')
    parser.add_argument('--discriminant-features',
                        default='results/feature_analysis/blast_annotations.tsv',
                        help='Path to discriminant features TSV (default: results/feature_analysis/blast_annotations.tsv)')
    parser.add_argument('--plot-unitig-taxonomy', action='store_true',
                        help='Create unitig PCA plots colored by taxonomy (requires BLAST annotations)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.sample_matrix and not args.sample_ids:
        parser.error("--sample-matrix requires --sample-ids")
    if args.sample_ids and not args.sample_matrix:
        parser.error("--sample-ids requires --sample-matrix")
    
    # Set defaults
    if args.output is None:
        args.output = Path(PATHS['figures_dir']) / 'pca_projection'
    
    print("=" * 80)
    print("PCA Projection of New Samples")
    print("=" * 80)
    
    # Load PCA reference
    reference_data = load_pca_reference(args.pca_model)
    pca_model = reference_data['pca_model']
    ref_sample_ids = reference_data['sample_ids']
    ref_unitig_ids = np.array(reference_data['unitig_ids'])
    scaler = reference_data.get('scaler', None)
    
    # Check for pre-computed reference coordinates
    if 'pca_coordinates' in reference_data:
        ref_pca = reference_data['pca_coordinates']
        logger.info(f"Using pre-computed reference PCA coordinates")
    else:
        logger.error("No pre-computed PCA coordinates in reference file!")
        logger.error("Please re-run 06_generate_pca_analysis.py to generate updated reference file")
        sys.exit(1)
    
    # Load reference metadata
    ref_metadata = pl.DataFrame(reference_data['metadata'])
    
    # Load new samples (from matrix files OR validation directories)
    if args.validation_dir:
        new_matrix, new_sample_ids, new_unitig_ids = load_from_validation_dirs(args.validation_dir)
    else:
        new_matrix, new_sample_ids, new_unitig_ids = load_new_samples(
            args.sample_matrix, 
            args.sample_ids
        )
    
    # Validate and align features
    aligned_matrix, alignment_info = validate_feature_alignment(
        new_matrix, 
        new_unitig_ids, 
        ref_unitig_ids
    )
    
    # Project new samples
    new_pca = project_samples(pca_model, aligned_matrix, scaler)
    
    # Compute similarity metrics
    similarity_metrics = compute_similarity_metrics(
        new_pca, 
        ref_pca, 
        ref_sample_ids, 
        k=args.k_neighbors
    )
    
    # Export results
    export_projection_results(
        new_pca, 
        new_sample_ids, 
        similarity_metrics, 
        args.output,
        ref_pca=ref_pca,
        ref_sample_ids=ref_sample_ids,
        export_reference=args.export_reference
    )
    
    # Load metadata for new samples if available
    new_metadata = load_validation_metadata(new_sample_ids)
    
    # Load BLAST annotations for plots
    blast_path = Path(args.blast_results)
    if not blast_path.exists():
        logger.warning(f"BLAST annotations not found: {blast_path}")
        logger.warning("Skipping feature evidence and taxonomy profile plots")
        logger.warning("To generate plots, run BLAST analysis first or provide --blast-results path")
        print("\n" + "=" * 80)
        print("Sample Projection Complete!")
        print("=" * 80)
        print(f"\nProjected {len(new_sample_ids)} samples")
        print(f"Output directory: {args.output}")
        print(f"\nNote: Skipped plots (BLAST annotations not found)")
        print(f"\nAlignment info:")
        for key, value in alignment_info.items():
            print(f"  {key}: {value}")
        return
    
    # Load BLAST results
    logger.info(f"Loading BLAST annotations from {blast_path}...")
    blast_df = pd.read_csv(
        blast_path,
        sep='\t',
        engine='python',
        header=None,
        names=['unitig_id', 'subject_id', 'pident', 'length', 'mismatch', 
               'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 
               'bitscore', 'taxid', 'description']
    )
    logger.info(f"  ✓ Loaded {len(blast_df)} BLAST annotations")
    
    # Load discriminant features
    feature_importance_path = Path(args.discriminant_features)
    if not feature_importance_path.exists():
        logger.warning(f"Discriminant features file not found: {feature_importance_path}")
        logger.warning("Using top 400 features by variance as fallback")
        # Fall back to top features by variance
        feature_variances = aligned_matrix.var(axis=0)
        top_400_indices = np.argsort(feature_variances)[-400:]
        discriminant_unitig_ids = ref_unitig_ids[top_400_indices].tolist()
    else:
        logger.info(f"Loading discriminant features from {feature_importance_path}...")
        feature_importance = pd.read_csv(feature_importance_path, sep='\t')
        discriminant_indices = feature_importance['feature_index'].unique()
        discriminant_unitig_ids = [ref_unitig_ids[idx] for idx in discriminant_indices if idx < len(ref_unitig_ids)]
        logger.info(f"  Discriminant feature_index range: {discriminant_indices.min()} to {discriminant_indices.max()}")
        logger.info(f"  Mapped to unitig_ids range: {min(discriminant_unitig_ids)} to {max(discriminant_unitig_ids)}")
    
    logger.info(f"\\nLoaded {len(discriminant_unitig_ids)} discriminant markers")
    
    # Create plots for each sample
    logger.info("\\n" + "="*80)
    logger.info("Creating Feature Evidence and Taxonomy Profile plots...")
    logger.info("="*80)
    
    for i, sample_id in enumerate(new_sample_ids):
        logger.info(f"\\n{'='*80}")
        logger.info(f"Processing {sample_id} ({i+1}/{len(new_sample_ids)})")
        logger.info(f"{'='*80}")
        
        sample_fractions = aligned_matrix[i, :]
        
        # Plot 1: Taxonomy Profile
        plot_taxonomy_profile(
            sample_id=sample_id,
            sample_unitig_fraction=sample_fractions,
            ref_unitig_ids=ref_unitig_ids,
            blast_annotations=blast_df,
            output_dir=args.output,
            threshold=0.0001,
            top_n=10
        )
        
        # Plot 3: Unitig PCA with taxonomy (optional)
        if args.plot_unitig_taxonomy:
            # Get PCA components from the model (shape: n_components × n_features)
            # Transpose to (n_features, n_components) for plotting
            ref_unitig_pca = pca_model.components_.T
            plot_unitig_pca_with_taxonomy(
                sample_id=sample_id,
                sample_unitig_fraction=sample_fractions,
                ref_unitig_pca=ref_unitig_pca,
                ref_unitig_ids=ref_unitig_ids,
                blast_annotations=blast_df,
                taxonomy_level='species',
                output_dir=args.output,
                top_n=10
            )
    
    print("\n" + "=" * 80)
    print("Sample Analysis Complete!")
    print("=" * 80)
    print(f"\nProcessed {len(new_sample_ids)} samples")
    print(f"Output directory: {args.output}")
    print(f"\nPlots created for each sample:")
    print(f"  - Feature Evidence (discriminant markers by phylum)")
    print(f"  - Taxonomy Profile (top 10 species)")
    print(f"\nAlignment info:")
    for key, value in alignment_info.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
