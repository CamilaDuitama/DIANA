#!/bin/bash
################################################################################
# Paper Figures and Tables Reproduction Script
################################################################################
#
# PURPOSE:
#   Regenerate all manuscript figures and tables from raw data.
#   Ensures complete reproducibility of published results.
#
# USAGE:
#   cd paper/
#   ./reproduce.sh
#
# REQUIREMENTS:
#   - DIANA environment installed (see ../environment.yml)
#   - Feature matrix at data/matrices/large_matrix_3070_with_frac/
#   - Metadata at data/metadata/DIANA_metadata.tsv
#
# OUTPUT:
#   - paper/figures/ - All manuscript figures (PNG + HTML)
#   - paper/tables/ - All manuscript tables (CSV + Markdown)
#
################################################################################

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root (one level up from paper/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "  DIANA Paper Reproduction Workflow"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "environment.yml" ]; then
    echo -e "${RED}ERROR: environment.yml not found${NC}"
    exit 1
fi

if [ ! -d "data/matrices/large_matrix_3070_with_frac" ]; then
    echo -e "${RED}ERROR: Feature matrix not found at data/matrices/large_matrix_3070_with_frac/${NC}"
    echo "Please ensure you have the unitig feature matrix before running this script."
    exit 1
fi

if [ ! -f "data/metadata/DIANA_metadata.tsv" ]; then
    echo -e "${RED}ERROR: Metadata file not found at data/metadata/DIANA_metadata.tsv${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# Determine Python command (prefer mamba environment)
if [ -d "./env" ]; then
    PYTHON_CMD="mamba run -p ./env python"
    echo "Using environment: ./env"
elif command -v mamba &> /dev/null && mamba env list | grep -q "diana"; then
    PYTHON_CMD="mamba run -n diana python"
    echo "Using environment: diana"
else
    PYTHON_CMD="python"
    echo -e "${YELLOW}WARNING: DIANA environment not found, using system Python${NC}"
fi
echo ""

# Create output directories
mkdir -p paper/figures paper/tables
echo -e "${GREEN}✓ Output directories created${NC}"
echo ""

#------------------------------------------------------------------------------
# STEP 1: Create Train/Test Splits
#------------------------------------------------------------------------------
echo "================================================================================"
echo "  STEP 1/4: Creating stratified train/test splits"
echo "================================================================================"
echo "Script: scripts/data_prep/01_create_splits.py"
echo "Input:  data/metadata/DIANA_metadata.tsv"
echo "Output: data/splits/{train,test,val}_ids.txt"
echo ""

# Skip if splits already exist (deterministic with random_state=42)
if [ -f "data/splits/train_ids.txt" ] && [ -f "data/splits/test_ids.txt" ]; then
    echo -e "${YELLOW}Splits already exist, skipping creation...${NC}"
    echo "To regenerate: rm -rf data/splits/ and re-run this script"
else
    $PYTHON_CMD scripts/data_prep/01_create_splits.py \
        --config configs/data_config.yaml
fi

echo ""
echo -e "${GREEN}✓ Step 1 complete${NC}"
echo ""

#------------------------------------------------------------------------------
# STEP 2: Analyze Unitig Features
#------------------------------------------------------------------------------
echo "================================================================================"
echo "  STEP 2/4: Analyzing unitig features"
echo "================================================================================"
echo "Script: scripts/data_prep/06_analyze_unitigs.py"
echo "Input:  data/matrices/large_matrix_3070_with_frac/unitigs.{fa,frac.mat,abundance.mat}"
echo "        data/splits/{train,test}_ids.txt (for split comparisons)"
echo "        data/metadata/DIANA_metadata.tsv (for PCA coloring)"
echo "Output: paper/figures/data_distribution/{frac,abundance}_mat/unitig_*.{png,html}"
echo "        paper/tables/{frac,abundance}_mat/unitig_*.csv"
echo ""
echo -e "${YELLOW}Note: Loading 1.6GB matrices takes ~1 minute each...${NC}"
echo ""

# Run for fraction matrix
echo "Analyzing FRACTION matrix..."
$PYTHON_CMD scripts/data_prep/06_analyze_unitigs.py \
    --matrix-dir data/matrices/large_matrix_3070_with_frac \
    --matrix-type frac \
    --metadata data/metadata/DIANA_metadata.tsv \
    --splits-dir data/splits \
    --output-figures paper/figures/data_distribution \
    --output-tables paper/tables

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Fraction matrix analysis failed${NC}"
    exit 1
fi

echo ""
echo "Analyzing ABUNDANCE matrix..."
$PYTHON_CMD scripts/data_prep/06_analyze_unitigs.py \
    --matrix-dir data/matrices/large_matrix_3070_with_frac \
    --matrix-type abundance \
    --metadata data/metadata/DIANA_metadata.tsv \
    --splits-dir data/splits \
    --output-figures paper/figures/data_distribution \
    --output-tables paper/tables

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Abundance matrix analysis failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Step 2 complete (both matrices analyzed)${NC}"
echo ""

#------------------------------------------------------------------------------
# STEP 3: Plot Metadata Distributions
#------------------------------------------------------------------------------
echo "================================================================================"
echo "  STEP 3/4: Plotting metadata distributions"
echo "================================================================================"
echo "Script: scripts/evaluation/02_plot_data_distribution.py"
echo "Input:  data/metadata/DIANA_metadata.tsv, data/splits/"
echo "Output: paper/figures/data_distribution/*.{png,html}"
echo ""

$PYTHON_CMD scripts/evaluation/02_plot_data_distribution.py \
    --config configs/data_config.yaml

echo ""
echo -e "${GREEN}✓ Step 3 complete${NC}"
echo ""

#------------------------------------------------------------------------------
# STEP 4: Statistical Tests for Split Balance
#------------------------------------------------------------------------------
echo "================================================================================"
echo "  STEP 4/4: Running statistical tests on splits"
echo "================================================================================"
echo "Script: scripts/evaluation/03_statistical_tests_splits.py"
echo "Input:  data/metadata/DIANA_metadata.tsv, data/splits/"
echo "Output: paper/tables/train_test_statistical_comparison.md"
echo ""

$PYTHON_CMD scripts/evaluation/03_statistical_tests_splits.py \
    --config configs/data_config.yaml

echo ""
echo -e "${GREEN}✓ Step 4 complete${NC}"
echo ""

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
echo "================================================================================"
echo -e "  ${GREEN}ALL STEPS COMPLETE${NC}"
echo "================================================================================"
echo ""
echo "Generated outputs:"
echo ""
echo "📊 Figures (paper/figures/):"
echo "  - unitig_length_distribution.{png,html}"
echo "  - unitig_gc_content.{png,html}"
echo "  - unitig_sparsity_distribution.{png,html}"
echo "  - unitig_length_vs_sparsity.{png,html}"
echo "  - data_distribution/*.{png,html}"
echo ""
echo "📋 Tables (paper/tables/):"
echo "  - unitig_sequence_stats.csv"
echo "  - unitig_sparsity_stats.csv"
echo "  - top20_common_unitigs.csv"
echo "  - train_test_statistical_comparison.md"
echo ""
echo "Interactive HTML plots available alongside PNG files."
echo "View them by opening *.html files in a web browser."
echo ""

# Count outputs
N_FIGURES=$(find paper/figures -name "*.png" | wc -l)
N_TABLES=$(find paper/tables -name "*.csv" -o -name "*.md" | wc -l)

echo "Summary: $N_FIGURES PNG figures + $N_TABLES tables generated"
echo ""
echo -e "${GREEN}✓ Paper reproduction complete!${NC}"
echo ""
echo "================================================================================"
