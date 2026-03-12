#!/usr/bin/env python3
"""
Statistical tests to compare train and test set distributions.

Performs Chi-squared tests for categorical variables and Kolmogorov-Smirnov
tests for numeric variables to verify splits maintain similar distributions.

Output: Markdown table in paper/tables/train_test_statistical_comparison.md
"""

import sys
import argparse
import logging
import polars as pl
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diana.data.loader import MetadataLoader
from diana.data.splitter import StratifiedSplitter
from diana.utils.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def chi_squared_test(train_series: pl.Series, test_series: pl.Series) -> Tuple[float, float, str]:
    """Perform Chi-squared test for categorical variables."""
    train_counts = train_series.value_counts()
    test_counts = test_series.value_counts()
    
    value_col, count_col = train_counts.columns[0], train_counts.columns[1]
    all_values = {v for v in set(train_counts[value_col].to_list()) | set(test_counts[value_col].to_list()) if v is not None}
    
    if len(all_values) == 0:
        return np.nan, np.nan, "No valid values"
    
    train_dict = dict(zip(train_counts[value_col].to_list(), train_counts[count_col].to_list()))
    test_dict = dict(zip(test_counts[value_col].to_list(), test_counts[count_col].to_list()))
    
    sorted_values = sorted(all_values, key=str)
    observed_train = [train_dict.get(val, 0) for val in sorted_values]
    observed_test = [test_dict.get(val, 0) for val in sorted_values]
    contingency = np.array([observed_train, observed_test])
    
    try:
        if contingency.shape[1] < 2:
            return np.nan, np.nan, "Only one category"
        
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        if (expected < 5).sum() > 0.2 * expected.size:
            if contingency.shape == (2, 2):
                _, p_value = stats.fisher_exact(contingency)
                return np.nan, p_value, f"Fisher's exact (p={p_value:.4f})"
            return chi2, p_value, "Chi-squared (low counts)"
        
        return chi2, p_value, "Chi-squared"
    except Exception as e:
        return np.nan, np.nan, f"Error: {str(e)}"


def ks_test(train_series: pl.Series, test_series: pl.Series) -> Tuple[float, float, str]:
    """Perform Kolmogorov-Smirnov test for continuous variables."""
    train_data = train_series.drop_nulls().to_numpy()
    test_data = test_series.drop_nulls().to_numpy()
    
    if len(train_data) < 3 or len(test_data) < 3:
        return np.nan, np.nan, "Insufficient data"
    
    try:
        statistic, p_value = stats.ks_2samp(train_data, test_data)
        return statistic, p_value, "Kolmogorov-Smirnov"
    except Exception as e:
        return np.nan, np.nan, f"Error: {str(e)}"


def add_unitig_tests(train_ids, test_ids, config):
    """Add statistical tests for unitig-level features (sparsity, GC content)."""
    from diana.data.loader import MatrixLoader
    from Bio import SeqIO
    from Bio.SeqUtils import gc_fraction
    
    results = []
    
    # Load matrix and sequences
    matrix_dir = Path(config.get("matrix_dir", "data/matrices/large_matrix_3070_with_frac"))
    unitigs_frac = matrix_dir / "unitigs.frac.mat"
    unitigs_fa = matrix_dir / "unitigs.fa"
    
    if not unitigs_frac.exists() or not unitigs_fa.exists():
        logger.warning(f"Unitig files not found in {matrix_dir}, skipping unitig tests")
        return results
    
    logger.info("Loading unitig matrix for split comparison...")
    loader = MatrixLoader(unitigs_frac)
    matrix, sample_ids, _ = loader.load()
    
    # Create sample masks
    train_mask = np.array([sid in train_ids for sid in sample_ids])
    test_mask = np.array([sid in test_ids for sid in sample_ids])
    
    train_matrix = matrix[train_mask]
    test_matrix = matrix[test_mask]
    
    # Test 1: Sparsity per sample (unitigs per sample)
    train_unitigs_per_sample = (train_matrix > 0).sum(axis=1)
    test_unitigs_per_sample = (test_matrix > 0).sum(axis=1)
    
    stat, pval, test_name = ks_test(
        pl.Series(train_unitigs_per_sample), 
        pl.Series(test_unitigs_per_sample)
    )
    
    results.append({
        "Variable": "Unitigs per Sample (Sparsity)",
        "Type": "Numeric",
        "Test": test_name,
        "Statistic": f"{stat:.4f}" if not np.isnan(stat) else "N/A",
        "P-value": f"{pval:.4f}" if not np.isnan(pval) else "N/A",
        "Significant": "Yes" if pval < 0.05 else "No" if not np.isnan(pval) else "N/A"
    })
    
    # Test 2: Mean fraction per sample
    train_mean_frac = train_matrix.mean(axis=1)
    test_mean_frac = test_matrix.mean(axis=1)
    
    stat, pval, test_name = ks_test(
        pl.Series(train_mean_frac),
        pl.Series(test_mean_frac)
    )
    
    results.append({
        "Variable": "Mean Fraction per Sample",
        "Type": "Numeric",
        "Test": test_name,
        "Statistic": f"{stat:.4f}" if not np.isnan(stat) else "N/A",
        "P-value": f"{pval:.4f}" if not np.isnan(pval) else "N/A",
        "Significant": "Yes" if pval < 0.05 else "No" if not np.isnan(pval) else "N/A"
    })
    
    # Test 3: GC content distribution (requires computing from sequences)
    logger.info("Computing GC content from unitig sequences...")
    gc_contents = []
    for record in SeqIO.parse(unitigs_fa, "fasta"):
        gc_contents.append(gc_fraction(record.seq) * 100)
    
    if len(gc_contents) > 0:
        # GC content is same for both splits, but test mean GC of present unitigs
        gc_array = np.array(gc_contents)
        
        # Get which unitigs are present in each split
        train_present = (train_matrix > 0).any(axis=0)  # Which unitigs present in train
        test_present = (test_matrix > 0).any(axis=0)   # Which unitigs present in test
        
        train_gc = gc_array[train_present]
        test_gc = gc_array[test_present]
        
        stat, pval, test_name = ks_test(
            pl.Series(train_gc),
            pl.Series(test_gc)
        )
        
        results.append({
            "Variable": "GC Content of Present Unitigs (%)",
            "Type": "Numeric",
            "Test": test_name,
            "Statistic": f"{stat:.4f}" if not np.isnan(stat) else "N/A",
            "P-value": f"{pval:.4f}" if not np.isnan(pval) else "N/A",
            "Significant": "Yes" if pval < 0.05 else "No" if not np.isnan(pval) else "N/A"
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Statistical tests for train/test split comparison')
    parser.add_argument('--config', default='configs/data_config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(Path(args.config))
    
    logger.info("Loading metadata...")
    metadata_path = Path(config["metadata_path"])
    if not metadata_path.exists():
        metadata_path = Path.cwd() / config["metadata_path"]
        
    df = MetadataLoader(metadata_path).load()
    
    # Cast numeric columns
    numeric_cols = [
        "publication_year", "total_runs_for_sample", "available_runs_for_sample", "unavailable_runs_for_sample",
        "unitig_size_gb", "Avg_read_len", "Avg_num_reads",
        "seqstats_contigs_n50", "seqstats_contigs_nbseq", "seqstats_contigs_maxlen", "seqstats_contigs_sumlen",
        "seqstats_unitigs_n50", "seqstats_unitigs_maxlen", "seqstats_unitigs_sumlen",
        "size_contigs_after_compression", "size_contigs_before_compression",
        "size_unitigs_before_compression", "size_unitigs_after_compression"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    
    logger.info(f"Total samples: {len(df)}")
    
    # Load splits
    splits_dir = Path(config["splits_dir"])
    logger.info("Loading splits...")
    train_ids, val_ids, test_ids = StratifiedSplitter.load_splits(splits_dir)
    
    id_col = "Run_accession"
    train_df = df.filter(pl.col(id_col).is_in(train_ids))
    test_df = df.filter(pl.col(id_col).is_in(test_ids))
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Test all columns except IDs
    columns_to_test = [col for col in df.columns if col not in ["Run_accession", "archive_accession"]]
    
    results = []
    logger.info("Performing statistical tests...")
    for col in columns_to_test:
        is_numeric = col in numeric_cols
        
        if is_numeric:
            train_non_null = train_df[col].drop_nulls().len()
            test_non_null = test_df[col].drop_nulls().len()
            
            if train_non_null < 3 or test_non_null < 3:
                results.append({"Variable": col, "Type": "Numeric", "Test": "N/A", "Statistic": "N/A", 
                              "P-value": "N/A", "Significant": "N/A", "Note": "Insufficient data"})
                continue
            
            statistic, p_value, test_name = ks_test(train_df[col], test_df[col])
        else:
            statistic, p_value, test_name = chi_squared_test(train_df[col], test_df[col])
        
        if np.isnan(p_value):
            significant, p_val_str, stat_str = "N/A", "N/A", "N/A"
        else:
            significant = "Yes" if p_value < 0.05 else "No"
            p_val_str = f"{p_value:.4f}"
            stat_str = f"{statistic:.4f}" if not np.isnan(statistic) else "N/A"
        
        results.append({"Variable": col, "Type": "Numeric" if is_numeric else "Categorical",
                       "Test": test_name, "Statistic": stat_str, "P-value": p_val_str,
                       "Significant": significant, "Note": "" if significant != "N/A" else test_name})
    
    # Add unitig-specific tests (sparsity and GC content)
    logger.info("Adding unitig-level statistical tests...")
    unitig_results = add_unitig_tests(train_ids, test_ids, config)
    results.extend(unitig_results)
    
    # Write results
    output_dir = Path("paper/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_test_statistical_comparison.md"
    
    logger.info(f"Writing results to {output_file}...")
    
    significant_count = sum(1 for r in results if r["Significant"] == "Yes")
    non_significant_count = sum(1 for r in results if r["Significant"] == "No")
    
    with open(output_file, 'w') as f:
        f.write("# Statistical Comparison of Train and Test Sets\n\n")
        f.write(f"**Total Samples**: {len(df)} (Train: {len(train_df)}, Test: {len(test_df)})\n\n")
        f.write("**Significance Level**: α = 0.05\n\n")
        f.write("**Test Methods**: Chi-squared for categorical, Kolmogorov-Smirnov for numeric\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Significant differences: {significant_count} / {len(results)}\n")
        f.write(f"- Non-significant: {non_significant_count} / {len(results)}\n\n")
        
        if significant_count > 0:
            f.write("⚠️ **Variables with significant differences**:\n")
            for r in results:
                if r["Significant"] == "Yes":
                    f.write(f"- **{r['Variable']}** (p = {r['P-value']})\n")
            f.write("\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Variable | Type | Test | Statistic | P-value | Significant | Note |\n")
        f.write("|----------|------|------|-----------|---------|-------------|------|\n")
        
        for r in results:
            f.write(f"| {r['Variable']} | {r['Type']} | {r['Test']} | {r['Statistic']} | "
                   f"{r['P-value']} | {r['Significant']} | {r.get('Note', '')} |\n")
    
    logger.info(f"✓ Results written. Significant differences in {significant_count} variables")

if __name__ == "__main__":
    main()
