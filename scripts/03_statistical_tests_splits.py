#!/usr/bin/env python3
"""Statistical tests to compare train and test set distributions."""

import sys
import argparse
import logging
import polars as pl
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def chi_squared_test(train_series: pl.Series, test_series: pl.Series) -> Tuple[float, float, str]:
    """
    Perform Chi-squared test for categorical variables.
    
    Returns:
        (statistic, p_value, interpretation)
    """
    # Get value counts
    train_counts = train_series.value_counts()
    test_counts = test_series.value_counts()
    
    # Get the column names (first column is the value, second is count)
    value_col = train_counts.columns[0]
    count_col = train_counts.columns[1]
    
    # Get all unique values
    all_values = set(train_counts[value_col].to_list()) | set(test_counts[value_col].to_list())
    
    # Remove None/null values from the set
    all_values = {v for v in all_values if v is not None}
    
    if len(all_values) == 0:
        return np.nan, np.nan, "No valid values"
    
    # Build contingency table
    train_dict = dict(zip(train_counts[value_col].to_list(), train_counts[count_col].to_list()))
    test_dict = dict(zip(test_counts[value_col].to_list(), test_counts[count_col].to_list()))
    
    # Sort values, converting to string for consistent sorting
    sorted_values = sorted(all_values, key=lambda x: str(x))
    
    observed_train = [train_dict.get(val, 0) for val in sorted_values]
    observed_test = [test_dict.get(val, 0) for val in sorted_values]
    
    # Contingency table
    contingency = np.array([observed_train, observed_test])
    
    try:
        if contingency.shape[1] < 2:
            return np.nan, np.nan, "Only one category"
        
        # Check for low expected frequencies
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Check if expected frequencies are too low
        if (expected < 5).sum() > 0.2 * expected.size:
            # Use Fisher's exact test for 2x2 tables
            if contingency.shape == (2, 2):
                _, p_value = stats.fisher_exact(contingency)
                return np.nan, p_value, f"Fisher's exact (p={p_value:.4f})"
            else:
                return chi2, p_value, f"Chi-squared (warning: low counts)"
        
        return chi2, p_value, "Chi-squared"
    except Exception as e:
        logger.warning(f"Chi-squared test failed: {e}")
        return np.nan, np.nan, f"Error: {str(e)}"


def ks_test(train_series: pl.Series, test_series: pl.Series) -> Tuple[float, float, str]:
    """
    Perform Kolmogorov-Smirnov test for continuous variables.
    
    Returns:
        (statistic, p_value, interpretation)
    """
    # Remove nulls
    train_data = train_series.drop_nulls().to_numpy()
    test_data = test_series.drop_nulls().to_numpy()
    
    if len(train_data) < 3 or len(test_data) < 3:
        return np.nan, np.nan, "Insufficient data"
    
    try:
        statistic, p_value = stats.ks_2samp(train_data, test_data)
        return statistic, p_value, "Kolmogorov-Smirnov"
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return np.nan, np.nan, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Statistical tests for train/test split comparison')
    parser.add_argument('--config', default='configs/data_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))
    
    # Load metadata
    logger.info("Loading metadata...")
    metadata_path = Path(config["metadata_path"])
    if not metadata_path.exists():
        metadata_path = Path.cwd() / config["metadata_path"]
        
    metadata_loader = MetadataLoader(metadata_path)
    df = metadata_loader.load()
    
    # Cast numeric columns explicitly
    numeric_cols = [
        "publication_year",
        "total_runs_for_sample", "available_runs_for_sample", "unavailable_runs_for_sample",
        "unitig_size_gb",
        "Avg_read_len", "Avg_num_reads",
        "seqstats_contigs_n50", "seqstats_contigs_nbseq", "seqstats_contigs_maxlen", "seqstats_contigs_sumlen",
        "seqstats_unitigs_n50", "seqstats_unitigs_maxlen", "seqstats_unitigs_sumlen",
        "size_contigs_after_compression", "size_contigs_before_compression",
        "size_unitigs_before_compression", "size_unitigs_after_compression"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
            except Exception as e:
                logger.warning(f"Could not cast {col} to float: {e}")
    
    logger.info(f"Total samples: {len(df)}")
    
    # Load splits
    splits_dir = Path(config["splits_dir"])
    logger.info("Loading splits...")
    train_ids, val_ids, test_ids = StratifiedSplitter.load_splits(splits_dir)
    
    # Subset data
    id_col = "Run_accession"
    train_df = df.filter(pl.col(id_col).is_in(train_ids))
    test_df = df.filter(pl.col(id_col).is_in(test_ids))
    
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Columns to test (exclude IDs)
    columns_to_test = [col for col in df.columns if col not in ["Run_accession", "archive_accession"]]
    
    # Perform tests
    results = []
    
    logger.info("Performing statistical tests...")
    for col in columns_to_test:
        logger.info(f"Testing {col}...")
        
        is_numeric = col in numeric_cols
        
        if is_numeric:
            # Check if we have enough non-null values
            train_non_null = train_df[col].drop_nulls().len()
            test_non_null = test_df[col].drop_nulls().len()
            
            if train_non_null < 3 or test_non_null < 3:
                results.append({
                    "Variable": col,
                    "Type": "Numeric",
                    "Test": "N/A",
                    "Statistic": "N/A",
                    "P-value": "N/A",
                    "Significant": "N/A",
                    "Note": "Insufficient data"
                })
                continue
            
            statistic, p_value, test_name = ks_test(train_df[col], test_df[col])
        else:
            # Categorical
            statistic, p_value, test_name = chi_squared_test(train_df[col], test_df[col])
        
        # Determine significance
        if np.isnan(p_value):
            significant = "N/A"
            p_val_str = "N/A"
            stat_str = "N/A"
        else:
            significant = "Yes" if p_value < 0.05 else "No"
            p_val_str = f"{p_value:.4f}"
            stat_str = f"{statistic:.4f}" if not np.isnan(statistic) else "N/A"
        
        results.append({
            "Variable": col,
            "Type": "Numeric" if is_numeric else "Categorical",
            "Test": test_name,
            "Statistic": stat_str,
            "P-value": p_val_str,
            "Significant": significant,
            "Note": "" if significant != "N/A" else test_name
        })
    
    # Create output directory
    output_dir = Path("paper/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown table
    output_file = output_dir / "train_test_statistical_comparison.md"
    
    logger.info(f"Writing results to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# Statistical Comparison of Train and Test Sets\n\n")
        f.write(f"**Total Samples**: {len(df)} (Train: {len(train_df)}, Test: {len(test_df)})\n\n")
        f.write("**Significance Level**: α = 0.05\n\n")
        f.write("**Test Methods**:\n")
        f.write("- **Categorical variables**: Chi-squared test (or Fisher's exact test for small counts)\n")
        f.write("- **Numeric variables**: Kolmogorov-Smirnov test (2-sample)\n\n")
        f.write("**Interpretation**:\n")
        f.write("- **Significant = Yes**: The distributions differ significantly (p < 0.05)\n")
        f.write("- **Significant = No**: No evidence of significant difference (p ≥ 0.05)\n\n")
        f.write("---\n\n")
        
        # Summary statistics
        significant_count = sum(1 for r in results if r["Significant"] == "Yes")
        non_significant_count = sum(1 for r in results if r["Significant"] == "No")
        na_count = sum(1 for r in results if r["Significant"] == "N/A")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Significant differences**: {significant_count} / {len(results)} variables\n")
        f.write(f"- **Non-significant**: {non_significant_count} / {len(results)} variables\n")
        f.write(f"- **Insufficient data / Not testable**: {na_count} / {len(results)} variables\n\n")
        
        if significant_count > 0:
            f.write("⚠️ **Variables with significant differences**:\n")
            for r in results:
                if r["Significant"] == "Yes":
                    f.write(f"- **{r['Variable']}** (p = {r['P-value']})\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## Detailed Results\n\n")
        
        # Table header
        f.write("| Variable | Type | Test | Statistic | P-value | Significant (α=0.05) | Note |\n")
        f.write("|----------|------|------|-----------|---------|----------------------|------|\n")
        
        # Table rows
        for r in results:
            note = r.get("Note", "")
            f.write(f"| {r['Variable']} | {r['Type']} | {r['Test']} | {r['Statistic']} | {r['P-value']} | {r['Significant']} | {note} |\n")
        
        f.write("\n")
    
    logger.info(f"Results written to {output_file}")
    logger.info(f"Significant differences found in {significant_count} variables")

if __name__ == "__main__":
    main()
