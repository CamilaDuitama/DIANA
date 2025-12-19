"""Matrix extraction and manipulation utilities."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import random

logger = logging.getLogger(__name__)

class MatrixExtractor:
    """Extract subsets of matrices for train/val/test splits."""
    
    def __init__(self, matrix_dir: Path):
        """
        Initialize extractor.
        
        Args:
            matrix_dir: Directory containing unitigs.pa.mat and unitigs.abundance.mat
        """
        self.matrix_dir = Path(matrix_dir)
        self.pa_file = self.matrix_dir / "unitigs.pa.mat"
        self.abundance_file = self.matrix_dir / "unitigs.abundance.mat"
        self.fof_path = self.matrix_dir / "kmer_matrix" / "kmtricks.fof"
        
    def validate_files(self) -> bool:
        """Check if required files exist."""
        files = [
            (self.pa_file, "PA matrix"),
            (self.abundance_file, "Abundance matrix"),
            (self.fof_path, "kmtricks.fof")
        ]
        
        all_exist = True
        for path, name in files:
            if not path.exists():
                logger.error(f"{name} not found: {path}")
                all_exist = False
        return all_exist
        
    def get_sample_order(self) -> List[str]:
        """Read sample order from kmtricks.fof."""
        sample_order = []
        with open(self.fof_path, 'r') as f:
            for line in f:
                if ':' in line:
                    sample_order.append(line.split(':')[0].strip())
        return sample_order
        
    def _validate_extraction(self, 
                           original_data: np.ndarray, 
                           extracted_data: np.ndarray, 
                           sample_mapping: Dict[int, int], 
                           unitig_ids: List[str], 
                           validation_samples: int = 50) -> bool:
        """Validate matrix extraction by comparing random samples."""
        n_samples, n_unitigs = extracted_data.shape
        validation_samples = min(validation_samples, n_samples)
        
        # Random sample indices
        sample_indices = random.sample(range(n_samples), validation_samples)
        unitig_indices = random.sample(range(n_unitigs), min(50, n_unitigs))
        
        errors = 0
        for sample_idx in sample_indices:
            orig_col = sample_mapping[sample_idx]
            for unitig_idx in unitig_indices:
                # -1 for 0-indexed access to original data (which is unitigs x samples)
                original_val = original_data[unitig_idx, orig_col - 1]
                extracted_val = extracted_data[sample_idx, unitig_idx]
                
                if original_val != extracted_val:
                    errors += 1
                    if errors <= 3:
                        logger.error(f"Mismatch at sample {sample_idx}, unitig {unitig_idx}: {original_val} != {extracted_val}")
        
        if errors == 0:
            return True
        else:
            logger.error(f"Found {errors} mismatches in validation")
            return False

    def _process_matrix(self, 
                       matrix_file: Path, 
                       matrix_type: str,
                       train_cols: List[int],
                       val_cols: List[int],
                       test_cols: List[int]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
        """Load and extract one matrix type."""
        logger.info(f"Processing {matrix_type} matrix...")
        
        # Load matrix
        matrix_data = []
        unitig_ids = []
        
        try:
            with open(matrix_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        unitig_ids.append(parts[0])
                        # Handle both int (PA) and float (abundance) values
                        if matrix_type == "PA":
                            matrix_data.append([int(x) if x.isdigit() else 0 for x in parts[1:]])
                        else:
                            matrix_data.append([float(x) if x.replace('.', '').replace('-', '').isdigit() else 0.0 for x in parts[1:]])
                            
            dtype = np.int8 if matrix_type == "PA" else np.float32
            matrix = np.array(matrix_data, dtype=dtype)
            logger.info(f"Matrix shape: {matrix.shape} ({matrix.nbytes / 1024**2:.1f} MB)")
            
            # Extract columns and transpose (samples x unitigs)
            # cols are 1-based in mapping, so subtract 1
            train_data = matrix[:, [c-1 for c in train_cols]].T
            val_data = matrix[:, [c-1 for c in val_cols]].T
            test_data = matrix[:, [c-1 for c in test_cols]].T
            
            return train_data, val_data, test_data, unitig_ids
            
        except Exception as e:
            logger.error(f"Error processing {matrix_type} matrix: {e}")
            return None

    def _write_matrix(self, 
                     data: np.ndarray, 
                     sample_ids: List[str], 
                     unitig_ids: List[str], 
                     output_path: Path, 
                     matrix_type: str):
        """Write matrix file."""
        with open(output_path, 'w') as f:
            f.write('sample_id ' + ' '.join(unitig_ids) + '\n')
            for i, sample_id in enumerate(sample_ids):
                if matrix_type == "PA":
                    row = [sample_id] + [str(int(x)) for x in data[i]]
                else:
                    row = [sample_id] + [f"{x:.6f}" if x != 0 else "0" for x in data[i]]
                f.write(' '.join(row) + '\n')

    def extract(self, 
                train_ids: List[str], 
                val_ids: List[str], 
                test_ids: List[str], 
                output_dir: Path) -> bool:
        """
        Extract train/val/test subsets from matrices.
        
        Args:
            train_ids: List of training sample IDs
            val_ids: List of validation sample IDs
            test_ids: List of test sample IDs
            output_dir: Output directory
            
        Returns:
            bool: True if successful
        """
        if not self.validate_files():
            return False
            
        sample_order = self.get_sample_order()
        sample_to_col = {sid: i + 1 for i, sid in enumerate(sample_order)}
        
        # Get column indices
        train_cols = [sample_to_col[sid] for sid in train_ids if sid in sample_to_col]
        val_cols = [sample_to_col[sid] for sid in val_ids if sid in sample_to_col]
        test_cols = [sample_to_col[sid] for sid in test_ids if sid in sample_to_col]
        
        # Check for missing samples
        all_ids = set(train_ids) | set(val_ids) | set(test_ids)
        missing = all_ids - set(sample_order)
        if missing:
            logger.warning(f"{len(missing)} samples not found in matrix: {list(missing)[:5]}...")
            
        if not (train_cols and val_cols and test_cols):
            logger.error("No valid samples found for one or more sets")
            return False
            
        # Process matrices
        pa_result = self._process_matrix(self.pa_file, "PA", train_cols, val_cols, test_cols)
        if not pa_result: return False
        train_pa, val_pa, test_pa, unitig_ids = pa_result
        
        abundance_result = self._process_matrix(self.abundance_file, "Abundance", train_cols, val_cols, test_cols)
        if not abundance_result: return False
        train_ab, val_ab, test_ab, _ = abundance_result
        
        # Get ordered sample IDs for output
        train_out_ids = [sample_order[c-1] for c in train_cols]
        val_out_ids = [sample_order[c-1] for c in val_cols]
        test_out_ids = [sample_order[c-1] for c in test_cols]
        
        # Write output
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Writing matrices...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self._write_matrix, train_pa, train_out_ids, unitig_ids, output_dir / 'train_matrix.pa.mat', "PA"),
                executor.submit(self._write_matrix, val_pa, val_out_ids, unitig_ids, output_dir / 'val_matrix.pa.mat', "PA"),
                executor.submit(self._write_matrix, test_pa, test_out_ids, unitig_ids, output_dir / 'test_matrix.pa.mat', "PA"),
                executor.submit(self._write_matrix, train_ab, train_out_ids, unitig_ids, output_dir / 'train_matrix.abundance.mat', "Abundance"),
                executor.submit(self._write_matrix, val_ab, val_out_ids, unitig_ids, output_dir / 'val_matrix.abundance.mat', "Abundance"),
                executor.submit(self._write_matrix, test_ab, test_out_ids, unitig_ids, output_dir / 'test_matrix.abundance.mat', "Abundance"),
            ]
            
            for future in futures:
                future.result()
                
        logger.info("Matrix extraction complete.")
        return True
