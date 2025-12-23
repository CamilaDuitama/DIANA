# Statistical Comparison of Train and Test Sets

**Total Samples**: 3070 (Train: 2609, Test: 461)

**Significance Level**: α = 0.05

**Test Methods**: Chi-squared for categorical, Kolmogorov-Smirnov for numeric

## Summary

- Significant differences: 1 / 49
- Non-significant: 46 / 49

⚠️ **Variables with significant differences**:
- **sample_host** (p = 0.0236)

## Detailed Results

| Variable | Type | Test | Statistic | P-value | Significant | Note |
|----------|------|------|-----------|---------|-------------|------|
| True_label | Categorical | Chi-squared (low counts) | 66.1420 | 0.1905 | No |  |
| sample_type | Categorical | Chi-squared | 0.0405 | 0.8404 | No |  |
| status | Categorical | Only one category | N/A | N/A | N/A | Only one category |
| sample_name | Categorical | Chi-squared (low counts) | 1761.8162 | 0.6223 | No |  |
| project_name | Categorical | Chi-squared (low counts) | 67.5222 | 0.8912 | No |  |
| publication_year | Numeric | Kolmogorov-Smirnov | 0.0338 | 0.7872 | No |  |
| geo_loc_name | Categorical | Chi-squared (low counts) | 49.8216 | 0.8227 | No |  |
| material | Categorical | Chi-squared | 6.4779 | 0.8901 | No |  |
| sample_host | Categorical | Chi-squared (low counts) | 22.0951 | 0.0236 | Yes |  |
| community_type | Categorical | Chi-squared | 0.1119 | 0.9998 | No |  |
| total_runs_for_sample | Numeric | Kolmogorov-Smirnov | 0.0411 | 0.5568 | No |  |
| available_runs_for_sample | Numeric | Kolmogorov-Smirnov | 0.0432 | 0.4936 | No |  |
| unavailable_runs_for_sample | Numeric | Kolmogorov-Smirnov | 0.0025 | 1.0000 | No |  |
| unitig_size_gb | Numeric | Kolmogorov-Smirnov | 0.0328 | 0.7793 | No |  |
| Assay Type | Categorical | Chi-squared (low counts) | 0.7601 | 0.6838 | No |  |
| BioProject | Categorical | Chi-squared (low counts) | 29.1009 | 0.4598 | No |  |
| BioSample | Categorical | Chi-squared (low counts) | 266.9592 | 0.7025 | No |  |
| Center Name | Categorical | Chi-squared (low counts) | 27.6731 | 0.2284 | No |  |
| Consent | Categorical | Only one category | N/A | N/A | N/A | Only one category |
| DATASTORE filetype | Categorical | Chi-squared (low counts) | 2.9233 | 0.7118 | No |  |
| DATASTORE provider | Categorical | Chi-squared (low counts) | 1.6177 | 0.8056 | No |  |
| DATASTORE region | Categorical | Chi-squared (low counts) | 2.4289 | 0.7872 | No |  |
| Experiment | Categorical | Chi-squared (low counts) | 318.0059 | 0.4894 | No |  |
| Instrument | Categorical | Chi-squared (low counts) | 9.6280 | 0.2107 | No |  |
| LibraryLayout | Categorical | Chi-squared | 0.8580 | 0.3543 | No |  |
| LibrarySelection | Categorical | Chi-squared (low counts) | 3.5083 | 0.6221 | No |  |
| LibrarySource | Categorical | Fisher's exact (p=0.1401) | N/A | 0.1401 | No |  |
| Organism | Categorical | Chi-squared (low counts) | 3.8716 | 0.8685 | No |  |
| Platform | Categorical | Fisher's exact (p=1.0000) | N/A | 1.0000 | No |  |
| ReleaseDate | Categorical | Chi-squared (low counts) | 37.1586 | 0.3256 | No |  |
| SRA Study | Categorical | Chi-squared (low counts) | 29.1009 | 0.4598 | No |  |
| Type | Categorical | Chi-squared | 0.3045 | 0.5811 | No |  |
| Isolation source | Categorical | Chi-squared (low counts) | 6.9423 | 0.4349 | No |  |
| Avg_read_len | Numeric | Kolmogorov-Smirnov | 0.0215 | 0.9914 | No |  |
| Avg_num_reads | Numeric | Kolmogorov-Smirnov | 0.0338 | 0.7453 | No |  |
| seqstats_contigs_n50 | Numeric | Kolmogorov-Smirnov | 0.0407 | 0.5509 | No |  |
| seqstats_contigs_nbseq | Numeric | Kolmogorov-Smirnov | 0.0271 | 0.9337 | No |  |
| seqstats_contigs_maxlen | Numeric | Kolmogorov-Smirnov | 0.0642 | 0.0874 | No |  |
| seqstats_contigs_sumlen | Numeric | Kolmogorov-Smirnov | 0.0391 | 0.5869 | No |  |
| seqstats_unitigs_n50 | Numeric | Kolmogorov-Smirnov | 0.0247 | 0.9699 | No |  |
| seqstats_unitigs_maxlen | Numeric | Kolmogorov-Smirnov | 0.0584 | 0.1432 | No |  |
| seqstats_unitigs_sumlen | Numeric | Kolmogorov-Smirnov | 0.0324 | 0.8022 | No |  |
| size_contigs_after_compression | Numeric | Kolmogorov-Smirnov | 0.0418 | 0.5174 | No |  |
| size_contigs_before_compression | Numeric | Kolmogorov-Smirnov | 0.0384 | 0.6235 | No |  |
| size_unitigs_before_compression | Numeric | Kolmogorov-Smirnov | 0.0328 | 0.7945 | No |  |
| size_unitigs_after_compression | Numeric | Kolmogorov-Smirnov | 0.0338 | 0.7603 | No |  |
| Unitigs per Sample (Sparsity) | Numeric | Kolmogorov-Smirnov | 0.0507 | 0.2557 | No |  |
| Mean Fraction per Sample | Numeric | Kolmogorov-Smirnov | 0.0472 | 0.3349 | No |  |
| GC Content of Present Unitigs (%) | Numeric | Kolmogorov-Smirnov | 0.0000 | 1.0000 | No |  |
