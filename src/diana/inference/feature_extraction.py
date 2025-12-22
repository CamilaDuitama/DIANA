"""
Feature extraction module for DIANA inference pipeline.

This module provides functionality to extract unitig-level feature vectors
from new metagenomic samples for downstream classification.
"""

import os
import subprocess
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DIANAFeatureExtractor:
    """
    Extract unitig abundance and fraction features from metagenomic samples.
    
    This class wraps the inference pipeline to convert raw FASTQ files into
    feature vectors compatible with trained DIANA models.
    
    Attributes:
        reference_dir: Directory containing MUSET training outputs
        k_size: K-mer size (must match training)
        min_abundance: Minimum k-mer count threshold (filters sequencing errors)
        threads: Number of CPU threads for parallel processing
    """
    
    def __init__(
        self,
        reference_dir: Union[str, Path],
        k_size: int = 31,
        min_abundance: int = 2,
        threads: int = 4
    ):
        """
        Initialize feature extractor.
        
        Args:
            reference_dir: Path to MUSET output directory containing:
                          - matrix.filtered.fasta (reference k-mer set)
                          - unitigs.fa (unitig sequences)
            k_size: K-mer size used during training
            min_abundance: Minimum k-mer count to consider present
                          (k-mers with count < min_abundance are set to 0)
            threads: Number of threads for back_to_sequences
        
        Raises:
            FileNotFoundError: If reference files don't exist
        """
        self.reference_dir = Path(reference_dir)
        self.k_size = k_size
        self.min_abundance = min_abundance
        self.threads = threads
        
        # Validate reference files exist
        self._validate_reference_files()
        
        # Locate pipeline scripts
        self.script_dir = self._find_script_dir()
        
        logger.info(f"Initialized DIANA feature extractor")
        logger.info(f"  Reference: {self.reference_dir}")
        logger.info(f"  K-mer size: {self.k_size}")
        logger.info(f"  Min abundance: {self.min_abundance}")
        logger.info(f"  Threads: {self.threads}")
    
    def _validate_reference_files(self):
        """Validate that required reference files exist."""
        required_files = [
            'matrix.filtered.fasta',
            'unitigs.fa'
        ]
        
        for filename in required_files:
            filepath = self.reference_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Required reference file not found: {filepath}\n"
                    f"Make sure you're pointing to a valid MUSET output directory."
                )
    
    def _find_script_dir(self) -> Path:
        """Locate the inference pipeline scripts directory."""
        # Assume scripts are in PROJECT_ROOT/scripts/inference
        current = Path(__file__).resolve()
        
        # Try to find project root by looking for setup.py or README.md
        for parent in current.parents:
            if (parent / 'setup.py').exists() or (parent / 'README.md').exists():
                script_dir = parent / 'scripts' / 'inference'
                if script_dir.exists():
                    return script_dir
        
        raise FileNotFoundError(
            "Could not locate inference pipeline scripts. "
            "Expected to find scripts/inference/ directory in project root."
        )
    
    def extract_features(
        self,
        fastq_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        cleanup: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract feature vectors from a single FASTQ sample.
        
        Args:
            fastq_path: Path to input FASTQ file
            output_dir: Directory for intermediate/output files
                       If None, uses temporary directory
            cleanup: Whether to remove intermediate files after extraction
        
        Returns:
            Dictionary containing:
                'abundance': np.ndarray of shape (n_unitigs,) - mean k-mer counts
                'fraction': np.ndarray of shape (n_unitigs,) - k-mer presence fractions
                'sample_name': str - sample identifier
                'n_unitigs': int - number of unitigs
        
        Raises:
            FileNotFoundError: If FASTQ file doesn't exist
            subprocess.CalledProcessError: If pipeline execution fails
        
        Example:
            >>> extractor = DIANAFeatureExtractor('/path/to/muset_output')
            >>> features = extractor.extract_features('sample.fastq')
            >>> abundance = features['abundance']  # Shape: (n_unitigs,)
            >>> fraction = features['fraction']    # Shape: (n_unitigs,)
        """
        fastq_path = Path(fastq_path)
        if not fastq_path.exists():
            raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")
        
        sample_name = fastq_path.stem
        
        # Set up output directory
        if output_dir is None:
            import tempfile
            output_dir = Path(tempfile.mkdtemp(prefix=f'diana_inference_{sample_name}_'))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting features for sample: {sample_name}")
        
        try:
            # Run inference pipeline
            self._run_pipeline(fastq_path, output_dir)
            
            # Load results
            abundance = self._load_vector(output_dir / f"{sample_name}_unitig_abundance.txt")
            fraction = self._load_vector(output_dir / f"{sample_name}_unitig_fraction.txt")
            
            # Validate dimensions match
            if len(abundance) != len(fraction):
                raise ValueError(
                    f"Dimension mismatch: abundance={len(abundance)}, fraction={len(fraction)}"
                )
            
            logger.info(f"✓ Extracted {len(abundance)} features")
            
            return {
                'abundance': abundance,
                'fraction': fraction,
                'sample_name': sample_name,
                'n_unitigs': len(abundance)
            }
        
        finally:
            # Cleanup intermediate files if requested
            if cleanup and output_dir.name.startswith('diana_inference_'):
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
    
    def extract_features_batch(
        self,
        fastq_paths: list,
        output_dir: Union[str, Path],
        parallel: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract features from multiple FASTQ samples.
        
        Args:
            fastq_paths: List of paths to FASTQ files
            output_dir: Directory for all outputs
            parallel: Whether to run back_to_sequences calls in parallel
                     (aggregation is still done individually per sample)
        
        Returns:
            Dictionary mapping sample_name -> features dict
            
        Example:
            >>> extractor = DIANAFeatureExtractor('/path/to/muset_output')
            >>> samples = ['sample1.fastq', 'sample2.fastq', 'sample3.fastq']
            >>> all_features = extractor.extract_features_batch(samples, 'output/')
            >>> for sample, features in all_features.items():
            ...     print(f"{sample}: {features['abundance'].shape}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Batch processing {len(fastq_paths)} samples")
        
        if parallel:
            # TODO: Implement parallel back_to_sequences + batch kmat_tools
            # For now, process sequentially
            logger.warning("Parallel batch processing not yet implemented, using sequential")
        
        results = {}
        for i, fastq_path in enumerate(fastq_paths, 1):
            logger.info(f"Processing {i}/{len(fastq_paths)}: {Path(fastq_path).name}")
            features = self.extract_features(
                fastq_path,
                output_dir=output_dir,
                cleanup=False  # Keep intermediate files for batch
            )
            results[features['sample_name']] = features
        
        logger.info(f"✓ Batch processing complete: {len(results)} samples")
        return results
    
    def _run_pipeline(self, fastq_path: Path, output_dir: Path):
        """Execute the inference pipeline."""
        # Create config file
        config_file = output_dir / 'pipeline_config.sh'
        with open(config_file, 'w') as f:
            f.write(f'SAMPLE_FASTQ="{fastq_path}"\n')
            f.write(f'MUSET_OUTPUT_DIR="{self.reference_dir}"\n')
            f.write(f'OUTPUT_DIR="{output_dir}"\n')
            f.write(f'K={self.k_size}\n')
            f.write(f'MIN_ABUNDANCE={self.min_abundance}\n')
            f.write(f'THREADS={self.threads}\n')
        
        # Run pipeline script
        pipeline_script = self.script_dir / 'inference_pipeline.sh'
        
        try:
            result = subprocess.run(
                ['bash', str(pipeline_script), str(config_file)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed: {e.stderr}")
            raise
    
    def _load_vector(self, filepath: Path) -> np.ndarray:
        """Load feature vector from text file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Output file not found: {filepath}")
        
        vector = np.loadtxt(filepath)
        return vector


def extract_diana_features(
    fastq_path: Union[str, Path],
    reference_dir: Union[str, Path],
    k_size: int = 31,
    min_abundance: int = 2,
    threads: int = 4,
    output_dir: Optional[Union[str, Path]] = None,
    return_both: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to extract features from a single sample.
    
    Args:
        fastq_path: Path to FASTQ file
        reference_dir: Path to MUSET output directory
        k_size: K-mer size (default: 31)
        min_abundance: Minimum k-mer count threshold (default: 2)
        threads: Number of CPU threads (default: 4)
        output_dir: Output directory (default: temporary)
        return_both: If True, returns (abundance, fraction)
                    If False, returns only abundance (default)
    
    Returns:
        If return_both=False: np.ndarray of shape (n_unitigs,) - abundance
        If return_both=True: tuple of (abundance, fraction) arrays
    
    Example:
        >>> # Get abundance features only
        >>> features = extract_diana_features('sample.fastq', 'muset_output/')
        >>> 
        >>> # Get both abundance and fraction
        >>> abundance, fraction = extract_diana_features(
        ...     'sample.fastq', 
        ...     'muset_output/',
        ...     return_both=True
        ... )
    """
    extractor = DIANAFeatureExtractor(
        reference_dir=reference_dir,
        k_size=k_size,
        min_abundance=min_abundance,
        threads=threads
    )
    
    result = extractor.extract_features(
        fastq_path=fastq_path,
        output_dir=output_dir
    )
    
    if return_both:
        return result['abundance'], result['fraction']
    else:
        return result['abundance']
