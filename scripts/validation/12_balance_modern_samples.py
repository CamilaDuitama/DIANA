#!/usr/bin/env python3
"""
Select balanced subset of modern samples matching target distribution.
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Target distribution
targets = {
    'human oral': 50,
    'human gut': 40,
    'soil': 30,
    'human skin': 20,
    'plant': 10
}

# Load expanded modern samples
df = pl.read_csv(
    PROJECT_ROOT / "data/validation/modern_samples_expanded.tsv",
    separator='\t'
)

print(f"📚 Loaded {len(df)} modern samples\n")

# Select balanced subset
selected = []

for search_term, target in targets.items():
    subset = df.filter(pl.col('search_term') == search_term)
    available = len(subset)
    
    if available >= target:
        # Randomly sample target number
        sampled = subset.sample(n=target, seed=42)
        selected.append(sampled)
        print(f"✅ {search_term:15s}: {target:3d} selected (from {available} available)")
    else:
        # Take all available
        selected.append(subset)
        print(f"⚠️  {search_term:15s}: {available:3d} selected (wanted {target}, all available)")

# Combine
final_df = pl.concat(selected)
print(f"\n📊 Total selected: {len(final_df)} modern samples")

# Save
output_file = PROJECT_ROOT / "data/validation/modern_samples_balanced.tsv"
final_df.write_csv(output_file, separator='\t')
print(f"💾 Saved to: {output_file}")

# Save accessions
acc_file = PROJECT_ROOT / "data/validation/modern_accessions_balanced.txt"
with open(acc_file, 'w') as f:
    f.write('\n'.join(final_df['run_accession'].to_list()))
print(f"💾 Saved {len(final_df)} accessions to: {acc_file}")

# Show study diversity
print(f"\n📊 Study diversity:")
study_dist = final_df.group_by(['search_term', 'study_name']).agg(
    pl.count('run_accession').alias('count')
).sort(['search_term', 'count'], descending=[False, True])

for term in targets.keys():
    term_studies = study_dist.filter(pl.col('search_term') == term)
    n_studies = len(term_studies)
    print(f"\n{term}: {n_studies} studies")
    print(term_studies.head(5))
