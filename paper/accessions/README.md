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
