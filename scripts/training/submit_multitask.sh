#!/bin/bash
################################################################################
# Multi-Task MLP Training Job Launcher
################################################################################
#
# PURPOSE:
#   Convenient launcher for multi-task training with preset configurations.
#   Simplifies submission of SLURM array jobs for different scenarios.
#
# DEPENDENCIES:
#   - SLURM workload manager
#   - scripts/training/run_multitask_gpu.sbatch
#   - Data files (matrix and metadata)
#
# USAGE:
#   ./scripts/training/submit_multitask.sh [test|prod|custom]
#
# MODES:
#   test   - Quick test on dummy data (100 samples, 2 folds, 5 trials, 20 epochs)
#            Good for: Verifying pipeline, testing changes, debugging
#            Runtime: ~2-3 minutes per fold
#   
#   prod   - Full production run (2609 samples, 5 folds, 50 trials, 200 epochs)
#            Good for: Final hyperparameter search, model selection
#            Runtime: ~2-4 hours per fold (depends on GPU)
#   
#   custom - Print instructions for manual configuration
#            Good for: Non-standard datasets, custom parameter ranges
#
# EXAMPLES:
#   # Test pipeline on dummy data:
#   ./scripts/training/submit_multitask.sh test
#
#   # Run full hyperparameter optimization:
#   ./scripts/training/submit_multitask.sh prod
#
#   # Custom dataset:
#   FEATURES=data/my_matrix.mat METADATA=data/my_metadata.tsv \\
#   TOTAL_FOLDS=3 N_TRIALS=10 MAX_EPOCHS=50 OUTPUT_DIR=results/custom \\
#   sbatch --array=0-2 scripts/training/run_multitask_gpu.sbatch
#
# OUTPUT:
#   - Job ID printed to stdout
#   - Monitor with: squeue -u $USER
#   - Check logs in: logs/multitask_gpu_*.out/err
#   - Results in: results/{OUTPUT_DIR}/fold_*/
#
################################################################################

# Launcher script for multi-task training with different configurations
# Usage: ./scripts/training/submit_multitask.sh [test|prod|custom]

MODE=${1:-prod}

case $MODE in
    test)
        echo "Submitting TEST job (dummy data: 100 samples, 2 folds, 5 trials, 20 epochs)..."
        echo "Usage: Set FEATURES and METADATA environment variables, or use defaults"
        FEATURES="${FEATURES:-data/test_data/splits/train_matrix_100feat.pa.mat}" \
        METADATA="${METADATA:-data/test_data/splits/train_metadata.tsv}" \
        OUTPUT_DIR="${OUTPUT_DIR:-results/multitask/hyperopt_test}" \
        LOG_DIR="${LOG_DIR:-logs/multitask/hyperopt_test}" \
        TOTAL_FOLDS=2 \
        N_TRIALS=5 \
        MAX_EPOCHS=20 \
        N_INNER_SPLITS=2 \
        sbatch --array=0-1 scripts/training/run_multitask_gpu.sbatch
        ;;
    
    prod)
        echo "Submitting PRODUCTION job (full data: 3070 samples, 5 folds, 50 trials, 200 epochs)..."
        echo "Set FEATURES and METADATA environment variables to specify input data"
        echo "Example: FEATURES=data/matrices/large_matrix_3070_with_frac/unitigs.frac.mat \\"
        echo "         METADATA=data/metadata/DIANA_metadata.tsv \\"
        echo "         ./scripts/training/submit_multitask.sh prod"
        
        if [ -z "$FEATURES" ] || [ -z "$METADATA" ]; then
            echo ""
            echo "ERROR: FEATURES and METADATA environment variables must be set for prod mode"
            echo "       This ensures you explicitly choose which matrix to use (.pa.mat, .frac.mat, etc.)"
            exit 1
        fi
        
        OUTPUT_DIR="${OUTPUT_DIR:-results/multitask/hyperopt}" \
        LOG_DIR="${LOG_DIR:-logs/multitask/hyperopt}" \
        TOTAL_FOLDS=5 \
        N_TRIALS=50 \
        MAX_EPOCHS=200 \
        N_INNER_SPLITS=3 \
        sbatch --array=0-4 scripts/training/run_multitask_gpu.sbatch
        ;;
    
    custom)
        echo "Submitting CUSTOM job with environment variables..."
        echo "Make sure to set: FEATURES, METADATA, OUTPUT_DIR, TOTAL_FOLDS, N_TRIALS, MAX_EPOCHS"
        echo "Example:"
        echo "  FEATURES=data/my_matrix.mat METADATA=data/my_metadata.tsv \\"
        echo "  TOTAL_FOLDS=3 N_TRIALS=10 MAX_EPOCHS=50 \\"
        echo "  sbatch --array=0-2 scripts/training/run_multitask_gpu.sbatch"
        ;;
    
    *)
        echo "Usage: $0 [test|prod|custom]"
        echo ""
        echo "Modes:"
        echo "  test   - Quick test on dummy data (100 samples, 2 folds, 5 trials)"
        echo "  prod   - Full production run (2609 samples, 5 folds, 50 trials)"
        echo "  custom - Instructions for custom configuration"
        exit 1
        ;;
esac
