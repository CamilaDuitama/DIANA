#!/bin/bash
#
# Generate All Paper Figures and Tables
#
# PURPOSE:
#     Master script to regenerate all publication-ready materials
#
# OUTPUTS:
#     - 4 main figures (confusion matrices, ROC/PR curves, feature importance, BLAST)
#     - 2 supplementary figures (runtime/memory, data split)
#     - 1 main table (performance summary)
#     - 1+ supplementary tables (class distribution, etc.)
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
run_script() {
    local script=$1
    local description=$2
    
    echo "→ $description"
    if python "$script" 2>&1; then
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

run_script "scripts/paper/generate_confusion_matrices.py" \
    "Main Figure 1: Confusion Matrices (4 tasks)"

run_script "scripts/paper/generate_roc_pr_curves.py" \
    "Main Figure 2: ROC and PR Curves (8 plots)"

run_script "scripts/paper/generate_feature_importance.py" \
    "Main Figure 3: Feature Importance (4 tasks)"

run_script "scripts/paper/generate_blast_hit_rate.py" \
    "Main Figure 4: BLAST Hit Rate Comparison"

# ============================================================================
# SUPPLEMENTARY FIGURES
# ============================================================================

echo "SUPPLEMENTARY FIGURES"
echo "---------------------"
echo ""

run_script "scripts/paper/generate_runtime_memory.py" \
    "Supplementary Figure 1: Runtime and Memory Scalability"

run_script "scripts/paper/generate_data_split.py" \
    "Supplementary Figure 2: Data Split Validation (6 plots)"

# ============================================================================
# MAIN TABLES
# ============================================================================

echo "MAIN TABLES"
echo "-----------"
echo ""

run_script "scripts/paper/generate_performance_summary_table.py" \
    "Main Table 1: Performance Summary"

run_script "scripts/paper/generate_computational_resources_table.py" \
    "Main Table 2: Computational Resources"

# ============================================================================
# SUPPLEMENTARY TABLES
# ============================================================================

echo "SUPPLEMENTARY TABLES"
echo "--------------------"
echo ""

run_script "scripts/paper/generate_class_distribution_table.py" \
    "Supplementary Table 1: Class Distribution"

run_script "scripts/paper/generate_unseen_labels_table.py" \
    "Supplementary Table 2: Unseen Labels"

run_script "scripts/paper/generate_perclass_performance_table.py" \
    "Supplementary Table 3: Per-Class Performance"

run_script "scripts/paper/generate_hyperparameters_table.py" \
    "Supplementary Table 4: Hyperparameters"

run_script "scripts/paper/generate_wrong_predictions_table.py" \
    "Supplementary Table 5: Wrong Predictions"

run_script "scripts/paper/generate_blast_summary_table.py" \
    "Supplementary Table 6: BLAST Summary"

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
