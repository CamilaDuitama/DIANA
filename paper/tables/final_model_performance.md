# DIANA Final Model Performance

Performance comparison across training (5-fold CV), test (held-out), and validation (external) datasets.

## Summary Table

| Task | Dataset | n | Accuracy (%) | Balanced Acc (%) | F1 Weighted (%) | Precision Macro (%) | Recall Macro (%) |
|------|---------|---|--------------|------------------|-----------------|---------------------|------------------|
| **Sample Type (ancient/modern)** | Training | 2,609 | 98.7 | 92.1 | 98.6 | 98.2 | 92.1 |
| | Test | 461 | **99.8** | **98.7** | **99.8** | **99.9** | **98.7** |
| | Validation (seen) | 608 | 93.1 | 93.1 | 96.4 | 50.0 | 93.1 |
| **Community Type** | Training | 2,609 | 97.5 | 97.7 | 97.5 | 98.5 | 97.7 |
| | Test | 461 | **98.9** | **99.6** | **98.9** | **98.0** | **99.6** |
| | Validation (seen) | 608 | 70.7 | 46.8 | 65.1 | 49.2 | 46.8 |
| **Sample Host (species)** | Training | 2,609 | 97.9 | 94.1 | 97.8 | 97.8 | 94.1 |
| | Test | 461 | **99.1** | **98.7** | **99.1** | **99.0** | **98.7** |
| | Validation (seen) | 538 | 80.9 | 63.4 | 83.5 | 61.2 | 63.4 |
| **Material** | Training | 2,609 | 97.5 | 94.2 | 97.4 | 95.8 | 94.2 |
| | Test | 461 | **98.7** | **90.9** | **98.6** | **88.4** | **90.9** |
| | Validation (seen) | 508 | 75.2 | 69.7 | 63.4 | 26.0 | 69.7 |

## Task Class Labels

- **Sample Type**: `Ancient`, `Modern`
- **Community Type**: `oral`, `gut`, `skeletal tissue`, `soft tissue`, `plant tissue`, `Not applicable - env sample`
- **Sample Host**: `Homo sapiens`, `Ursus arctos`, `Arabidopsis thaliana`, `Ambrosia artemisiifolia`, `Pan troglodytes`, `Gorilla sp.`, and others (12-20 species)
- **Material**: `dental calculus`, `tooth`, `bone`, `sediment`, `soft_tissue`, `digestive_contents`, `leaf`, and others (13-20 types)

## Notes

- **Training**: Performance on full training set (2,609 samples) after training
- **Test**: Held-out test set (461 samples, 0% overlap with training)
- **Validation (seen)**: External validation on AncientMetagenomeDir v25.09.0, showing only samples with labels seen during training. Precision/Recall macro not computed for validation.
- **Balanced Accuracy**: Accounts for class imbalance by averaging per-class recall
- **F1 Weighted**: Weighted average of per-class F1 scores (weights = class frequencies)
- **Precision/Recall Macro**: Unweighted average across all classes

## Key Findings

1. **Near-perfect training performance**: 98.7-99.8% accuracy on full training set indicates model has learned the training data well
2. **Excellent generalization**: Test performance remains high despite not seeing those samples during training
3. **Domain shift on validation**: 10-30% performance drop for biological tasks on external data from different sources
4. **Sample type robustness**: Cannot evaluate on validation (all samples are 'ancient', 'modern' label never seen)
5. **Improved validation with seen labels**: Material 62.5%→74.9%, Sample Host 71.7%→80.8% when excluding unseen classes
