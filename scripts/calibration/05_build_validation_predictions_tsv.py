#!/usr/bin/env python3
"""
Build consolidated validation_predictions.tsv from individual JSON prediction files.
"""

import json
import pandas as pd
from pathlib import Path


def load_label_encoders(encoders_path):
    """Load label encoders - format is {task: {classes: [...]}}"""
    with open(encoders_path) as f:
        encoders = json.load(f)
    # Convert to {task: [class_list]} for easier lookup
    return {task: data['classes'] for task, data in encoders.items()}


def build_consolidated_predictions():
    """Build consolidated TSV file from individual prediction JSONs."""
    
    # Paths
    predictions_dir = Path('results/validation_predictions')
    metadata_file = Path('paper/metadata/validation_metadata.tsv')
    label_encoders_path = Path('results/full_training/label_encoders.json')
    output_file = Path('results/validation_predictions/validation_predictions.tsv')
    
    print(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file, sep='\t')
    print(f"  Loaded {len(metadata)} samples")
    
    print(f"\nLoading label encoders from {label_encoders_path}...")
    encoders = load_label_encoders(label_encoders_path)
    for task, classes in encoders.items():
        print(f"  {task}: {len(classes)} classes")
    
    # Find all prediction JSON files
    print(f"\nScanning {predictions_dir} for prediction files...")
    prediction_files = list(predictions_dir.glob('*/*_predictions.json'))
    print(f"  Found {len(prediction_files)} prediction files")
    
    results = []
    failed = []
    
    print("\nProcessing predictions...")
    for i, pred_file in enumerate(prediction_files):
        sample_id = pred_file.parent.name
        
        # Show progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(prediction_files)}...")
        
        try:
            # Check if sample has SUCCESS status
            jobinfo_file = pred_file.parent / f'{sample_id}.jobinfo'
            if jobinfo_file.exists():
                with open(jobinfo_file) as f:
                    jobinfo = json.load(f)
                    if jobinfo.get('status') != 'SUCCESS':
                        failed.append((sample_id, 'job_failed'))
                        continue
            
            # Load prediction JSON
            with open(pred_file) as f:
                pred_data = json.load(f)
            
            # Find sample in metadata (use Run_accession column)
            sample_meta = metadata[metadata['Run_accession'] == sample_id]
            if len(sample_meta) == 0:
                failed.append((sample_id, 'not_in_metadata'))
                continue
            
            sample_meta = sample_meta.iloc[0]
            
            # Build row
            row = {'sample_id': sample_id}
            
            # Process each task
            for task in ['sample_type', 'community_type', 'sample_host', 'material']:
                if task not in pred_data['predictions']:
                    failed.append((sample_id, f'missing_{task}'))
                    break
                
                task_pred = pred_data['predictions'][task]
                
                # Get predicted class using index
                pred_idx = task_pred['class_index']
                pred_class = encoders[task][pred_idx]
                
                # Get true label from metadata
                true_label = sample_meta[task]
                
                # Add predictions and labels
                row[f'{task}_pred'] = pred_class
                row[f'{task}_true'] = true_label
                row[f'{task}_confidence'] = task_pred['confidence']
                
                # Add probabilities for all classes
                probs = task_pred['probabilities']
                for idx, class_name in enumerate(encoders[task]):
                    # Probabilities might be keyed by string index or class name
                    if str(idx) in probs:
                        prob = probs[str(idx)]
                    elif class_name in probs:
                        prob = probs[class_name]
                    else:
                        # Handle abbreviated names (e.g., "ancient" vs "ancient_metagenome")
                        abbreviated = class_name.replace('_metagenome', '')
                        prob = probs.get(abbreviated, 0.0)
                    
                    row[f'{task}_prob_{class_name}'] = prob
            else:
                # All tasks processed successfully
                results.append(row)
                continue
            
        except Exception as e:
            failed.append((sample_id, f'exception: {str(e)}'))
            continue
    
    # Create DataFrame
    print(f"\nBuilding DataFrame from {len(results)} successful predictions...")
    
    if len(results) == 0:
        print("\nERROR: No successful predictions!")
        print("\nFailed samples:")
        for sample_id, reason in failed[:20]:
            print(f"  {sample_id}: {reason}")
        return
    
    df = pd.DataFrame(results)
    
    # Save to TSV
    output_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nSuccess!")
    print(f"  Saved: {output_file}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed samples (showing first 10):")
        for sample_id, reason in failed[:10]:
            print(f"  {sample_id}: {reason}")


if __name__ == '__main__':
    build_consolidated_predictions()
