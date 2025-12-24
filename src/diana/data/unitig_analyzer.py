"""
Unitig Sequence Analysis Utilities
===================================

Provides efficient tools for loading and analyzing unitig sequences from FASTA files.

CLASSES:
--------
UnitigAnalyzer: Load sequences, compute statistics, extract features

USAGE:
------
# Load sequences and compute basic stats
analyzer = UnitigAnalyzer('data/matrices/large_matrix_3070_with_frac/unitigs.fa')
stats_df = analyzer.compute_sequence_stats()

# Extract specific sequences by indices
sequences = analyzer.get_sequences_by_indices([0, 1, 2, 100, 500])

# Get sequences for top features
top_features = [1234, 5678, 9012]
feature_seqs = analyzer.get_feature_sequences(top_features)
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import polars as pl
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
import logging

logger = logging.getLogger(__name__)


class UnitigAnalyzer:
    """
    Efficient loading and analysis of unitig sequences.
    
    Uses seqkit for fast sequence extraction and BioPython for statistics.
    Caches sequences in memory for repeated access.
    """
    
    def __init__(self, fasta_path: str):
        """
        Initialize unitig analyzer.
        
        Args:
            fasta_path: Path to unitigs.fa file
        """
        self.fasta_path = Path(fasta_path)
        self._sequences = None  # Lazy loading
        self._ids = None
        self._index_map = None  # Map from sequence ID to index
        
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")
    
    def _load_sequences(self):
        """Load all sequences into memory (called lazily)."""
        if self._sequences is not None:
            return  # Already loaded
        
        logger.info(f"Loading unitig sequences from {self.fasta_path}...")
        self._sequences = []
        self._ids = []
        
        for record in SeqIO.parse(self.fasta_path, "fasta"):
            self._sequences.append(str(record.seq))
            self._ids.append(record.id)
        
        # Create index map (ID -> position in list)
        self._index_map = {seq_id: idx for idx, seq_id in enumerate(self._ids)}
        
        logger.info(f"Loaded {len(self._sequences)} unitig sequences")
    
    def get_sequences_by_indices(self, indices: List[int]) -> Dict[int, str]:
        """
        Get sequences for specific feature indices.
        
        Args:
            indices: List of feature indices (0-based, matching matrix columns)
            
        Returns:
            Dict mapping index -> sequence
        """
        self._load_sequences()
        
        sequences = {}
        for idx in indices:
            if 0 <= idx < len(self._sequences):
                sequences[idx] = self._sequences[idx]
            else:
                logger.warning(f"Index {idx} out of range (0-{len(self._sequences)-1})")
        
        return sequences
    
    def get_sequences_by_ids(self, seq_ids: List[str]) -> Dict[str, str]:
        """
        Get sequences for specific sequence IDs.
        
        Args:
            seq_ids: List of sequence IDs from FASTA headers
            
        Returns:
            Dict mapping ID -> sequence
        """
        self._load_sequences()
        
        sequences = {}
        for seq_id in seq_ids:
            if seq_id in self._index_map:
                idx = self._index_map[seq_id]
                sequences[seq_id] = self._sequences[idx]
            else:
                logger.warning(f"Sequence ID {seq_id} not found")
        
        return sequences
    
    def get_all_sequences(self) -> Tuple[List[str], List[str]]:
        """
        Get all sequences and their IDs.
        
        Returns:
            Tuple of (ids, sequences)
        """
        self._load_sequences()
        return self._ids, self._sequences
    
    def compute_sequence_stats(self) -> pl.DataFrame:
        """
        Compute statistics for all unitig sequences.
        
        Returns:
            Polars DataFrame with columns: index, id, length, gc_content, complexity
        """
        self._load_sequences()
        
        logger.info("Computing sequence statistics...")
        
        stats_list = []
        for idx, (seq_id, seq) in enumerate(zip(self._ids, self._sequences)):
            # Compute GC content
            gc = gc_fraction(seq) * 100
            
            # Compute sequence complexity (Shannon entropy)
            from collections import Counter
            base_counts = Counter(seq.upper())
            total = len(seq)
            entropy = 0.0
            for count in base_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            stats_list.append({
                'index': idx,
                'id': seq_id,
                'length': len(seq),
                'gc_content': gc,
                'complexity': entropy,
                'n_count': seq.upper().count('N')
            })
        
        df = pl.DataFrame(stats_list)
        logger.info(f"Computed statistics for {len(df)} sequences")
        
        return df
    
    def extract_sequences_seqkit(
        self, 
        indices: List[int], 
        output_fasta: Optional[Path] = None
    ) -> str:
        """
        Extract sequences using seqkit (fast for large subsets).
        
        Args:
            indices: List of 0-based indices to extract
            output_fasta: Optional output path for FASTA file
            
        Returns:
            FASTA formatted string
        """
        self._load_sequences()
        
        # Convert indices to sequence IDs
        seq_ids = [self._ids[idx] for idx in indices if 0 <= idx < len(self._ids)]
        
        if not seq_ids:
            logger.warning("No valid indices provided")
            return ""
        
        # Create temporary file with IDs
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_id_file = f.name
            for seq_id in seq_ids:
                f.write(f"{seq_id}\n")
        
        try:
            # Use seqkit to extract sequences
            cmd = ['seqkit', 'grep', '-f', temp_id_file, str(self.fasta_path)]
            
            if output_fasta:
                cmd.extend(['-o', str(output_fasta)])
                subprocess.run(cmd, check=True, capture_output=True)
                logger.info(f"Extracted {len(seq_ids)} sequences to {output_fasta}")
                with open(output_fasta, 'r') as f:
                    return f.read()
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                return result.stdout
        
        except subprocess.CalledProcessError as e:
            logger.error(f"seqkit failed: {e.stderr}")
            # Fallback to BioPython
            return self._extract_with_biopython(seq_ids)
        
        except FileNotFoundError:
            logger.warning("seqkit not found, using BioPython fallback")
            return self._extract_with_biopython(seq_ids)
        
        finally:
            # Clean up temp file
            Path(temp_id_file).unlink(missing_ok=True)
    
    def _extract_with_biopython(self, seq_ids: List[str]) -> str:
        """Fallback: extract sequences using BioPython."""
        sequences = self.get_sequences_by_ids(seq_ids)
        
        fasta_str = ""
        for seq_id, seq in sequences.items():
            fasta_str += f">{seq_id}\n{seq}\n"
        
        return fasta_str
    
    def compute_stats_for_indices(self, indices: List[int]) -> pl.DataFrame:
        """
        Compute statistics for specific feature indices.
        
        Args:
            indices: List of feature indices
            
        Returns:
            DataFrame with statistics for selected sequences
        """
        # Get all stats first
        all_stats = self.compute_sequence_stats()
        
        # Filter to requested indices
        selected_stats = all_stats.filter(pl.col('index').is_in(indices))
        
        return selected_stats.sort('index')
