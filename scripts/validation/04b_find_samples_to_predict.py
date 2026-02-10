#!/usr/bin/env python3
"""
Find samples that need predictions (don't have predictions.json yet).
Creates a filtered metadata file and array indices file.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Load metadata
metadata_file = PROJECT_ROOT / "paper/metadata/validation_metadata.tsv"
df = pl.read_csv(metadata_file, separator='\t')

print(f"Total samples in metadata: {len(df)}")

# Check which samples already have predictions
predictions_dir = PROJECT_ROOT / "results/validation_predictions"
samples_needing_prediction = []
samples_already_predicted = []

for i, row in enumerate(df.iter_rows(named=True), 1):
    run_acc = row['run_accession']
    prediction_file = predictions_dir / run_acc / f"{run_acc}_predictions.json"
    
    if prediction_file.exists():
        samples_already_predicted.append((i, run_acc))
    else:
        samples_needing_prediction.append((i, run_acc))

print(f"Already predicted: {len(samples_already_predicted)}")
print(f"Need prediction: {len(samples_needing_prediction)}")

if len(samples_needing_prediction) == 0:
    print("\n✅ All samples already have predictions!")
    exit(0)

# Save filtered metadata (only samples needing prediction)
indices_to_keep = [idx - 1 for idx, _ in samples_needing_prediction]  # Convert to 0-based
df_filtered = df[indices_to_keep]

output_metadata = PROJECT_ROOT / "data/validation/samples_to_predict.tsv"
df_filtered.write_csv(output_metadata, separator='\t')
print(f"\n✅ Saved filtered metadata: {output_metadata}")
print(f"   Contains {len(df_filtered)} samples needing prediction")

# Save array indices for SLURM (1-based, comma-separated for --array)
if len(samples_needing_prediction) <= 20:
    # If small number, list them explicitly
    array_spec = ",".join([str(idx) for idx, _ in samples_needing_prediction])
else:
    # If large, use range
    array_spec = f"1-{len(samples_needing_prediction)}"

array_file = PROJECT_ROOT / "data/validation/array_indices.txt"
with open(array_file, 'w') as f:
    f.write(array_spec + '\n')

print(f"✅ Saved array indices: {array_file}")
print(f"   Use: sbatch --array={array_spec}%10")

# Show first few samples
print(f"\nFirst 5 samples to predict:")
for idx, run_acc in samples_needing_prediction[:5]:
    print(f"  {idx}: {run_acc}")

if len(samples_needing_prediction) > 5:
    print(f"  ... and {len(samples_needing_prediction) - 5} more")
