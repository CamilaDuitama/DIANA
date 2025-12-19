"""
Feature Extractor for Diana Inference

Computes unitig-based features for new aDNA samples using the same methodology
as MUSET training data (fraction of k-mers present in each unitig).

This ensures consistency between training and inference feature distributions.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    unitigs_fasta: Path
    """Path to unitigs.fa file from MUSET training run."""
    
    sshash_dict: Path
    """Path to unitigs.sshash.dict file (SSHash k-mer dictionary)."""
    
    kmer_size: int = 31
    """K-mer size used in training (default: 31)."""
    
    min_kmer_abundance: int = 2
    """Minimum k-mer abundance threshold (default: 2)."""
    
    temp_dir: Optional[Path] = None
    """Temporary directory for intermediate files."""


class FeatureExtractor:
    """
    Extract unitig-based features from new aDNA samples.
    
    Uses SSHash dictionary to map k-mers to unitigs and computes
    the fraction of k-mers present in each unitig (matching training methodology).
    
    Example:
        >>> config = FeatureExtractionConfig(
        ...     unitigs_fasta=Path("unitigs.fa"),
        ...     sshash_dict=Path("unitigs.sshash.dict")
        ... )
        >>> extractor = FeatureExtractor(config)
        >>> features = extractor.extract_from_fastq("sample.fastq.gz")
    """
    
    def __init__(self, config: FeatureExtractionConfig):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration with paths to unitigs and SSHash dictionary.
        
        Raises:
            FileNotFoundError: If required files don't exist.
        """
        self.config = config
        self._validate_files()
        self._load_unitig_metadata()
        
    def _validate_files(self):
        """Validate that all required files exist."""
        if not self.config.unitigs_fasta.exists():
            raise FileNotFoundError(f"Unitigs FASTA not found: {self.config.unitigs_fasta}")
        
        if not self.config.sshash_dict.exists():
            raise FileNotFoundError(f"SSHash dictionary not found: {self.config.sshash_dict}")
            
    def _load_unitig_metadata(self):
        """Load unitig metadata (lengths, IDs) from FASTA file."""
        logger.info(f"Loading unitig metadata from {self.config.unitigs_fasta}")
        
        self.unitig_ids = []
        self.unitig_lengths = []
        self.unitig_num_kmers = []
        
        with open(self.config.unitigs_fasta, 'r') as f:
            current_id = None
            current_seq = []
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous unitig
                    if current_id is not None:
                        seq = ''.join(current_seq)
                        seq_len = len(seq)
                        num_kmers = max(0, seq_len - self.config.kmer_size + 1)
                        
                        self.unitig_ids.append(current_id)
                        self.unitig_lengths.append(seq_len)
                        self.unitig_num_kmers.append(num_kmers)
                    
                    # Start new unitig
                    current_id = line[1:].split()[0]  # Remove '>' and take first token
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last unitig
            if current_id is not None:
                seq = ''.join(current_seq)
                seq_len = len(seq)
                num_kmers = max(0, seq_len - self.config.kmer_size + 1)
                
                self.unitig_ids.append(current_id)
                self.unitig_lengths.append(seq_len)
                self.unitig_num_kmers.append(num_kmers)
        
        self.num_unitigs = len(self.unitig_ids)
        logger.info(f"Loaded {self.num_unitigs} unitigs")
        
    def extract_from_fastq(
        self, 
        fastq_path: Union[str, Path],
        sample_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract unitig features from a FASTQ file.
        
        Pipeline:
        1. Count k-mers using Jellyfish
        2. Query SSHash dictionary for each k-mer -> get unitig ID
        3. For each unitig, compute fraction of k-mers present
        4. Return feature vector (length = num_unitigs)
        
        Args:
            fastq_path: Path to FASTQ file (can be gzipped).
            sample_id: Optional sample identifier for logging.
            
        Returns:
            Feature vector of shape (num_unitigs,) with fraction values [0, 1].
        """
        fastq_path = Path(fastq_path)
        if not fastq_path.exists():
            raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")
        
        sample_id = sample_id or fastq_path.stem
        logger.info(f"Extracting features for sample: {sample_id}")
        
        # Step 1: Count k-mers with Jellyfish
        kmer_counts = self._count_kmers(fastq_path)
        
        # Step 2: Query SSHash and compute fractions
        features = self._compute_unitig_fractions(kmer_counts)
        
        logger.info(f"Extracted features: {len(features)} unitigs, "
                   f"{np.sum(features > 0)} non-zero")
        
        return features
    
    def _count_kmers(self, fastq_path: Path) -> Dict[str, int]:
        """
        Count k-mers using Jellyfish.
        
        Args:
            fastq_path: Path to FASTQ file.
            
        Returns:
            Dictionary mapping k-mer -> abundance.
        """
        # TODO: Implement Jellyfish k-mer counting
        # For now, placeholder that will use Jellyfish or kmtricks
        raise NotImplementedError(
            "K-mer counting not yet implemented. "
            "Will use Jellyfish or kmtricks in final version."
        )
    
    def _compute_unitig_fractions(self, kmer_counts: Dict[str, int]) -> np.ndarray:
        """
        Compute fraction of k-mers present for each unitig.
        
        Uses SSHash dictionary to map k-mers to unitigs.
        
        Args:
            kmer_counts: Dictionary of k-mer -> abundance.
            
        Returns:
            Array of fractions, shape (num_unitigs,).
        """
        # TODO: Implement SSHash querying via Python bindings or subprocess
        # For now, placeholder
        raise NotImplementedError(
            "SSHash querying not yet implemented. "
            "Will use Python bindings or C++ wrapper in final version."
        )


def extract_features_from_fastq(
    fastq_path: Union[str, Path],
    unitigs_fasta: Union[str, Path],
    sshash_dict: Union[str, Path],
    **kwargs
) -> np.ndarray:
    """
    Convenience function to extract features from a FASTQ file.
    
    Args:
        fastq_path: Path to FASTQ file.
        unitigs_fasta: Path to unitigs.fa file.
        sshash_dict: Path to unitigs.sshash.dict file.
        **kwargs: Additional arguments passed to FeatureExtractionConfig.
        
    Returns:
        Feature vector of shape (num_unitigs,).
    """
    config = FeatureExtractionConfig(
        unitigs_fasta=Path(unitigs_fasta),
        sshash_dict=Path(sshash_dict),
        **kwargs
    )
    
    extractor = FeatureExtractor(config)
    return extractor.extract_from_fastq(fastq_path)
