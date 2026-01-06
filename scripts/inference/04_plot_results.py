#!/usr/bin/env python3
"""
04_plot_results.py

Generate interactive barplots of predicted label probabilities using Plotly.

Usage:
    python 04_plot_results.py --predictions predictions.json --output_dir OUTPUT_DIR

- predictions.json: Output from 03_run_inference.py
- OUTPUT_DIR: Directory to save plots (one per task)
"""
import argparse
import json
import os
from pathlib import Path
import plotly.graph_objects as go

def load_label_encoders(model_dir: str = "results/training") -> dict:
    """Load label encoders to map numeric labels to names."""
    encoders_path = Path(model_dir) / "label_encoders.json"
    if encoders_path.exists():
        with open(encoders_path, 'r') as f:
            return json.load(f)
    return {}

def plot_predictions(predictions, output_dir, sample_id, label_encoders=None):
    # Extract the predictions dict from the full output
    if "predictions" in predictions:
        predictions = predictions["predictions"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for task, task_preds in predictions.items():
        # Get probabilities from the nested structure
        if "probabilities" in task_preds:
            probs_dict = task_preds["probabilities"]
        else:
            probs_dict = task_preds
        
        # Map numeric labels to names if available
        if label_encoders and task in label_encoders:
            class_list = label_encoders[task]["classes"]
            # Map string indices to actual class names
            labels = []
            probs = []
            for idx_str, prob in probs_dict.items():
                try:
                    idx = int(idx_str)
                    if idx < len(class_list):
                        labels.append(class_list[idx])
                    else:
                        labels.append(idx_str)
                except (ValueError, TypeError):
                    # Already a string label name
                    labels.append(idx_str)
                probs.append(prob)
        else:
            labels = list(probs_dict.keys())
            probs = [probs_dict[label] for label in labels]
        
        # Get predicted class info
        predicted_class = task_preds.get("predicted_class", "unknown")
        confidence = task_preds.get("confidence", 0.0)
        
        # Create plotly bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=probs,
                marker_color='skyblue',
                text=[f'{p:.3f}' for p in probs],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f'{task.replace("_", " ").title()}<br><sub>Sample: {sample_id} | Predicted: {predicted_class} (confidence: {confidence:.3f})</sub>',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Class',
            yaxis_title='Predicted Probability',
            yaxis=dict(range=[0, 1.05]),
            template='plotly_white',
            height=500,
            margin=dict(b=150),  # Extra margin for rotated labels
            xaxis=dict(tickangle=-45)
        )
        
        # Save as interactive HTML and static PNG
        output_html = os.path.join(output_dir, f'{sample_id}_{task}_barplot.html')
        output_png = os.path.join(output_dir, f'{sample_id}_{task}_barplot.png')
        
        fig.write_html(output_html)
        print(f"Interactive plot saved: {output_html}")
        
        try:
            fig.write_image(output_png, width=800, height=500)
            print(f"Static plot saved: {output_png}")
        except Exception as e:
            print(f"Note: Could not save static PNG (kaleido may not be installed): {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot prediction results')
    parser.add_argument('--predictions', required=True, help='Path to predictions JSON file')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    parser.add_argument('--sample_id', default='sample', help='Sample identifier')
    parser.add_argument('--label_encoders', default='results/training', help='Path to directory containing label_encoders.json')
    args = parser.parse_args()
    
    # Load predictions
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    # Extract sample_id from predictions if available
    sample_id = predictions.get('sample_id', args.sample_id)
    
    # Load label encoders
    label_encoders = load_label_encoders(args.label_encoders)
    
    # Generate plots
    plot_predictions(predictions, args.output_dir, sample_id, label_encoders)

if __name__ == '__main__':
    main()
