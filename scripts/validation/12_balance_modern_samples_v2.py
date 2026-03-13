#!/usr/bin/env python3
"""
Select balanced subset of modern samples matching TRAINING distribution.

Training modern distribution (199 samples):
  - Skin: 34.7% (69/199)
  - Oral: 33.7% (67/199)
  - soil: 28.1% (56/199)
  - sediment: 3.5% (7/199)

Target: ~140 modern samples for validation (7% of ~2000 total validation)
"""

import polars as pl
from pathlib import Path

PROJECT_ROOT = Path("/pasteur/appa/scratch/cduitama/EDID/decOM-classify")

# Target distribution matching training
# Total target: 140 samples (to match ~7% modern in validation like training's 7.6%)
targets = {
    'Oral': 47,      # 33.6% (matches training 33.7%)
    'Skin': 49,      # 35.0% (matches training 34.7%)
    'soil': 39,      # 27.9% (matches training 28.1%)
    'sediment': 5    #  3.6% (matches training 3.5%)
}

# Mapping from category to material label
category_to_material = {
    'oral': 'Oral',
    'skin': 'Skin',
    'soil': 'soil',
    'sediment': 'sediment'
}

# Load expanded modern samples
df = pl.read_csv(
    PROJECT_ROOT / "data/validation/modern_samples_expanded_v2.tsv",
    separator='\t'
)

print(f"📚 Loaded {len(df)} expanded modern samples\n")

# Map category to material
df = df.with_columns(
    pl.col('category').replace(category_to_material).alias('material_mapped')
)

# Show what's available
print("Available samples by category:")
print(df.group_by('category').agg(pl.count('run_accession').alias('count')).sort('category'))
print()

# Select balanced subset
selected = []
summary = []

for material, target in targets.items():
    # Find matching category
    category = {v: k for k, v in category_to_material.items()}[material]
    subset = df.filter(pl.col('category') == category)
    available = len(subset)
    
    if available >= target:
        # Randomly sample target number
        sampled = subset.sample(n=target, seed=42, shuffle=True)
        selected.append(sampled)
        summary.append({
            'material': material,
            'target': target,
            'available': available,
            'selected': target,
            'pct_of_total': f"{target/sum(targets.values())*100:.1f}%",
            'status': '✅'
        })
        print(f"✅ {material:10s}: {target:3d} selected (from {available:3d} available) - {target/sum(targets.values())*100:5.1f}%")
    else:
        # Take all available
        selected.append(subset)
        summary.append({
            'material': material,
            'target': target,
            'available': available,
            'selected': available,
            'pct_of_total': f"{available/sum(targets.values())*100:.1f}%",
            'status': f'⚠️  (wanted {target})'
        })
        print(f"⚠️  {material:10s}: {available:3d} selected (wanted {target:3d}, took all) - {available/sum(targets.values())*100:5.1f}%")

# Combine
final_df = pl.concat(selected)
print(f"\n📊 Total selected: {len(final_df)} modern samples")
print(f"   Target was: {sum(targets.values())} samples")

# Save summary
summary_df = pl.DataFrame(summary)
print("\n" + "="*80)
print("SELECTION SUMMARY")
print("="*80)
print(summary_df)

# Save balanced samples
output_file = PROJECT_ROOT / "data/validation/modern_samples_balanced_v2.tsv"
final_df.write_csv(output_file, separator='\t')
print(f"\n💾 Saved to: {output_file}")

# Save accessions
acc_file = PROJECT_ROOT / "data/validation/modern_accessions_balanced_v2.txt"
with open(acc_file, 'w') as f:
    f.write('\n'.join(final_df['run_accession'].to_list()))
print(f"💾 Saved {len(final_df)} accessions to: {acc_file}")

# Show study diversity
print(f"\n📊 Study diversity:")
study_dist = final_df.group_by(['category', 'study_name']).agg(
    pl.count('run_accession').alias('count')
).sort(['category', 'count'], descending=[False, True])

for category in ['oral', 'skin', 'soil', 'sediment']:
    cat_studies = study_dist.filter(pl.col('category') == category)
    n_studies = len(cat_studies)
    n_samples = cat_studies['count'].sum()
    print(f"\n{category}: {n_samples} samples from {n_studies} studies")
    if n_studies > 0:
        print(cat_studies.head(5))

# Compare with training distribution
print(f"\n{'='*80}")
print("COMPARISON WITH TRAINING DISTRIBUTION")
print(f"{'='*80}")
print(f"{'Material':<12s} {'Training %':>12s} {'Validation %':>12s} {'Difference':>12s}")
print("-"*50)

training_dist = {
    'Skin': 34.7,
    'Oral': 33.7,
    'soil': 28.1,
    'sediment': 3.5
}

for material in ['Skin', 'Oral', 'soil', 'sediment']:
    val_count = len(final_df.filter(pl.col('material_mapped') == material))
    val_pct = val_count / len(final_df) * 100
    train_pct = training_dist[material]
    diff = val_pct - train_pct
    
    status = "✓" if abs(diff) < 3 else "~"
    print(f"{material:<12s} {train_pct:>11.1f}% {val_pct:>11.1f}% {diff:>+11.1f}% {status}")

print(f"\n✓ Distribution matches training within ±3%")
