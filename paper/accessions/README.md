# DIANA Dataset Accessions

This directory contains SRA accession numbers for all datasets used in the DIANA study.

## Files

### `train_accessions.txt`
- **N = 2,608** SRA run accessions
- **Purpose**: Training the multi-task neural network
- **Source**: Curated from [AncientMetagenomeDir](https://github.com/SPAAM-community/AncientMetagenomeDir) database
- **Notes**: These samples are used to learn the relationships between unitig features and sample characteristics

### `test_accessions.txt`
- **N = 460** SRA run accessions
- **Purpose**: Held-out test set for final model evaluation
- **Source**: Same dataset as training, randomly split (stratified by sample characteristics)
- **Notes**: Never seen during training; used to assess generalization performance

### `validation_accessions.txt`
- **N = 616** SRA run accessions
- **Purpose**: Independent validation on completely external dataset
- **Source**: AncientMetagenomeDir v25.09.0 samples not included in training/test
- **Notes**: 100% ancient metagenomes, 100% host-associated samples. Used to evaluate model performance on truly independent data from different studies and geographic locations.

## Dataset Characteristics

| Dataset | N | Sample Type | Community Type | Geographic Coverage |
|---------|---|-------------|----------------|---------------------|
| **Training** | 2,608 | Ancient + Modern | All types | Global |
| **Test** | 460 | Ancient + Modern | All types | Global |
| **Validation** | 616 | 100% Ancient | 100% Host-associated | 28 countries |

## Data Availability

All samples are publicly available from the NCBI Sequence Read Archive (SRA):
- **Access**: https://www.ncbi.nlm.nih.gov/sra
- **Method**: Download using SRA accession numbers provided in these files
- **Tools**: Use `prefetch` and `fasterq-dump` from [SRA Toolkit](https://github.com/ncbi/sra-tools)

### Example Usage

```bash
# Download a single sample
prefetch SRR13263117
fasterq-dump SRR13263117

# Download all training samples
cat train_accessions.txt | xargs -I {} prefetch {}

# Download all validation samples
cat validation_accessions.txt | xargs -I {} prefetch {}
```

## Full Metadata

Complete metadata for each dataset (including sample characteristics, geographic origin, publication references) is available in:
- Training/Test: `data/metadata/DIANA_metadata.tsv`
- Validation: `data/validation/validation_metadata_expanded.tsv`

## Citation

If you use these datasets, please cite:
1. **DIANA manuscript** (this study) - [citation pending]
2. **AncientMetagenomeDir**: Fellows Yates et al. (2021). Community-curated and standardised metadata of published ancient metagenomic samples with AncientMetagenomeDir. *Scientific Data* 8, 31. https://doi.org/10.1038/s41597-021-00816-y

## Contact

For questions about this dataset:
- Open an issue at: https://github.com/CamilaDuitama/DIANA/issues
