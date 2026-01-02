#!/usr/bin/env python3
"""
04_plot_results.py

Generate barplots of predicted label probabilities for each task from the model inference output.

Usage:
    python 04_plot_results.py --predictions predictions.json --output_dir OUTPUT_DIR

- predictions.json: Output from 03_run_inference.py
- OUTPUT_DIR: Directory to save plots (one per task)
"""
import argparse
import json
import os
import matplotlib.pyplot as plt

def plot_predictions(predictions, output_dir, sample_id):
    for task, task_preds in predictions.items():
        labels = list(task_preds.keys())
        probs = [task_preds[label] for label in labels]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(labels, probs, color='skyblue')
        plt.ylabel('Predicted Probability')
        plt.xlabel('Label')
        plt.title(f'{task} - {sample_id}')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        for bar, prob in zip(bars, probs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{prob:.2f}',
                     ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{sample_id}_{task}_barplot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Plot saved: {plot_path}')

def main():
    parser = argparse.ArgumentParser(description='Plot predicted label probabilities per task.')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--output_dir', required=True, help='Directory to save plots')
    parser.add_argument('--sample_id', default=None, help='Sample ID (optional, overrides JSON)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.predictions) as f:
        data = json.load(f)
    # Support both {task: {label: prob}} and {"sample_id": ..., "predictions": {...}}
    if 'predictions' in data:
        predictions = data['predictions']
        sample_id = args.sample_id or data.get('sample_id', 'sample')
    else:
        predictions = data
        sample_id = args.sample_id or 'sample'
    plot_predictions(predictions, args.output_dir, sample_id)

if __name__ == '__main__':
    main()
