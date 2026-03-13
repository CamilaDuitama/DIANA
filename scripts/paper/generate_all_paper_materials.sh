#!/bin/bash
#
# Generate All Paper Figures and Tables
#
# PURPOSE:
#     Master script to regenerate all publication-ready materials
#
# OUTPUTS:
#     - 3 main figures (confusion matrices, ROC/PR curves, feature importance)
#     - 6 supplementary figures (runtime/memory, data split, PCA, PCA loadings, Logan search, baseline comparison)
#     - 2 main tables (performance summary, computational resources)
#     - 9 supplementary tables (class distribution, unseen labels, per-class perf,
#       hyperparameters, wrong predictions, BLAST, seen/unseen validation,
#       matrix generation, BioProject offenders)
#
# USAGE:
#     bash scripts/paper/generate_all_paper_materials.sh
#
# AUTHOR: Paper generation pipeline

echo "================================================================================"
echo "GENERATING ALL PAPER FIGURES AND TABLES"
echo "================================================================================"
echo ""

# Track success
GENERATED=0
FAILED=0

# Function to run a script and check status
PYTHON="${PYTHON:-./env/bin/python}"

run_script() {
    local script=$1
    local description=$2
    
    echo "→ $description"
    if "$PYTHON" "$script" 2>&1; then
        ((GENERATED++))
        echo "  ✓ Success"
        return 0
    else
        ((FAILED++))
        echo "  ✗ Failed"
        return 1
    fi
    echo ""
}

# ============================================================================
# MAIN FIGURES
# ============================================================================

echo "MAIN FIGURES"
echo "------------"
echo ""

run_script "scripts/paper/12_generate_confusion_matrices.py" \
    "Main Figure 1: Confusion Matrices (4 tasks)"

run_script "scripts/paper/13_generate_roc_pr_curves.py" \
    "Main Figure 2: ROC and PR Curves (8 plots)"

run_script "scripts/paper/14_generate_feature_importance.py" \
    "Main Figure 3: Feature Importance (4 tasks)"

# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================

echo "SUPPLEMENTARY FIGURES"
echo "---------------------"
echo ""

run_script "scripts/paper/16_generate_runtime_memory.py" \
    "Supplementary Figure 1: Runtime and Memory Scalability"

run_script "scripts/paper/02_generate_data_split.py" \
    "Supplementary Figure 2: Data Split Validation (6 plots)"

run_script "scripts/paper/06_generate_pca_analysis.py" \
    "Supplementary Figure 3 & 4: PCA Analysis and PCA Loading Plots"

run_script "scripts/paper/20_generate_logan_search_taxonomy_barplot.py" \
    "Supplementary Figure 5: Logan Database Coverage and Taxonomy"

run_script "scripts/paper/21_generate_baseline_comparison.py" \
    "Supplementary Figure 6: Baseline Comparison"

# ============================================================================
# MAIN TABLES
# ============================================================================

echo "MAIN TABLES"
echo "-----------"
echo ""

run_script "scripts/paper/03_generate_performance_summary_table.py" \
    "Main Table 1: Performance Summary"

run_script "scripts/paper/11_generate_computational_resources_table.py" \
    "Main Table 2: Computational Resources"

# ============================================================================
# SUPPLEMENTARY TABLES
# ============================================================================

echo "SUPPLEMENTARY TABLES"
echo "--------------------"
echo ""

run_script "scripts/paper/01_generate_class_distribution_table.py" \
    "Supplementary Table 1: Class Distribution"

run_script "scripts/paper/05_generate_unseen_labels_table.py" \
    "Supplementary Table 2: Unseen Labels"

run_script "scripts/paper/04_generate_perclass_performance_table.py" \
    "Supplementary Table 3: Per-Class Performance"

run_script "scripts/paper/08_generate_hyperparameters_table.py" \
    "Supplementary Table 4: Hyperparameters"

run_script "scripts/paper/09_generate_wrong_predictions_table.py" \
    "Supplementary Table 5: Wrong Predictions"

run_script "scripts/paper/10_generate_blast_summary_table.py" \
    "Supplementary Table 6: BLAST Summary"

run_script "scripts/paper/17_generate_seen_unseen_table.py" \
    "Supplementary Table 7: Seen vs Unseen Validation Performance"

run_script "scripts/paper/18_generate_matrix_generation_table.py" \
    "Supplementary Table 8: Feature Matrix Generation Parameters"

run_script "scripts/paper/26_generate_bioproject_offenders_table.py" \
    "Supplementary Table 9: BioProject Error Sources (offenders)"

# ============================================================================
# SUMMARY
# ============================================================================

echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "  Generated: $GENERATED scripts"
echo "  Failed:    $FAILED scripts"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ ALL PAPER MATERIALS GENERATED SUCCESSFULLY"
    echo ""
    echo "Outputs:"
    echo "  Figures: paper/figures/final/"
    echo "  Tables:  paper/tables/final/"
    echo ""
    
    # Count generated files
    PNG_COUNT=$(find paper/figures/final -name "*.png" 2>/dev/null | wc -l)
    HTML_COUNT=$(find paper/figures/final -name "*.html" 2>/dev/null | wc -l)
    TEX_COUNT=$(find paper/tables/final -name "*.tex" 2>/dev/null | wc -l)
    
    echo "  PNG files:   $PNG_COUNT"
    echo "  HTML files:  $HTML_COUNT"
    echo "  LaTeX files: $TEX_COUNT"
    
    exit 0
else
    echo "⚠ SOME SCRIPTS FAILED - CHECK ERRORS ABOVE"
    exit 1
fi
