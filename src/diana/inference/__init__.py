"""
Diana Inference Module

This module handles prediction for new aDNA samples.
Feature extraction is performed by the shell pipeline in scripts/inference/:
  01_count_kmers.sh        - back_to_sequences: count reference k-mers in sample
  02_aggregate_to_unitigs.sh - kmat_tools unitig: aggregate to unitig-level vectors
  03_run_inference.py      - load unitig fraction vector and run the neural network
"""

__all__ = []
