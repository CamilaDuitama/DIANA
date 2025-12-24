#!/usr/bin/env python3
"""
Generate all publication-quality interactive plots and tables using Plotly.

Creates both HTML (interactive) and PNG (static) versions of all visualizations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plotly theme configuration
TEMPLATE = 'plotly_white'
COLOR_PALETTE = px.colors.qualitative.Set2


def load_data(metrics_path: Path, history_path: Path, config_path: Path, 
              predictions_path: Path, encoders_path: Path):
    """Load all required data files."""
    logger.info("Loading data files...")
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(encoders_path, 'r') as f:
        encoders = json.load(f)
    
    predictions_df = pd.read_csv(predictions_path, sep='\t')
    
    return metrics, history, config, predictions_df, encoders


def save_figure(fig, output_path: Path, width=1200, height=800):
    """Save figure as both HTML and PNG."""
    # Save HTML (interactive)
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    logger.info(f"Saved HTML: {html_path}")
    
    # Save PNG (static)
    png_path = output_path.with_suffix('.png')
    fig.write_image(str(png_path), width=width, height=height, scale=2)
    logger.info(f"Saved PNG: {png_path}")


def plot_multitask_performance_summary(metrics: Dict, output_path: Path):
    """Multi-task performance comparison bar chart."""
    logger.info("Creating multi-task performance summary...")
    
    tasks = list(metrics.keys())
    metric_names = ['accuracy', 'f1_macro', 'balanced_accuracy']
    metric_labels = ['Accuracy', 'F1-Macro', 'Balanced Accuracy']
    
    fig = go.Figure()
    
    for i, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        values = [metrics[task][metric_name] for task in tasks]
        fig.add_trace(go.Bar(
            name=metric_label,
            x=[t.replace('_', ' ').title() for t in tasks],
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            marker_color=COLOR_PALETTE[i]
        ))
    
    fig.update_layout(
        title='Multi-Task Model Performance (Test Set)',
        xaxis_title='Task',
        yaxis_title='Score',
        barmode='group',
        template=TEMPLATE,
        yaxis_range=[0.9, 1.02],
        font=dict(size=14),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    save_figure(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         task_name: str, output_path: Path):
    """Interactive confusion matrix heatmap."""
    logger.info(f"Creating confusion matrix for {task_name}...")
    
    # Calculate percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create hover text
    hover_text = []
    for i in range(len(class_names)):
        hover_row = []
        for j in range(len(class_names)):
            hover_row.append(
                f'True: {class_names[i]}<br>'
                f'Predicted: {class_names[j]}<br>'
                f'Count: {cm[i, j]}<br>'
                f'Percentage: {cm_norm[i, j]:.1f}%'
            )
        hover_text.append(hover_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=[[f'{cm[i,j]}<br>({cm_norm[i,j]:.1f}%)' 
               for j in range(len(class_names))] 
              for i in range(len(class_names))],
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertext=hover_text,
        hoverinfo='text',
        colorscale='Blues',
        colorbar=dict(title='Count')
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {task_name.replace("_", " ").title()} (Test Set)',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        template=TEMPLATE,
        font=dict(size=12),
        width=max(600, len(class_names) * 80),
        height=max(600, len(class_names) * 80)
    )
    
    save_figure(fig, output_path, 
                width=max(800, len(class_names) * 100),
                height=max(800, len(class_names) * 100))


def plot_per_class_metrics(classification_report: Dict, task_name: str, 
                          output_path: Path):
    """Per-class precision, recall, F1-score bar chart."""
    logger.info(f"Creating per-class metrics for {task_name}...")
    
    classes = [k for k in classification_report.keys() 
               if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics_data = {
        'Precision': [classification_report[cls]['precision'] for cls in classes],
        'Recall': [classification_report[cls]['recall'] for cls in classes],
        'F1-Score': [classification_report[cls]['f1-score'] for cls in classes]
    }
    
    fig = go.Figure()
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        fig.add_trace(go.Bar(
            name=metric,
            x=classes,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            marker_color=COLOR_PALETTE[i]
        ))
    
    fig.update_layout(
        title=f'Per-Class Metrics - {task_name.replace("_", " ").title()} (Test Set)',
        xaxis_title='Class',
        yaxis_title='Score',
        barmode='group',
        template=TEMPLATE,
        yaxis_range=[0, 1.1],
        font=dict(size=12),
        xaxis_tickangle=-45,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    save_figure(fig, output_path,
                width=max(1000, len(classes) * 120),
                height=700)


def plot_training_curves(history: Dict, output_path: Path):
    """Training and validation loss curves."""
    logger.info("Creating training/validation loss curves...")
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['train_loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Mark best epoch if available
    if 'best_epoch' in history:
        best_epoch = history['best_epoch']
        fig.add_vline(
            x=best_epoch,
            line_dash='dash',
            line_color='green',
            annotation_text=f'Best Epoch ({best_epoch})',
            annotation_position='top'
        )
    
    fig.update_layout(
        title='Training and Validation Loss (Training Set)',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template=TEMPLATE,
        font=dict(size=14),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    save_figure(fig, output_path)


def plot_roc_curves(predictions_df: pd.DataFrame, task_name: str,
                   class_names: List[str], output_path: Path):
    """ROC curves for multi-class classification."""
    logger.info(f"Creating ROC curves for {task_name}...")
    
    y_true = predictions_df[f'{task_name}_true_idx'].values
    
    # Get probability columns
    prob_cols = [f'{task_name}_prob_{i}' for i in range(len(class_names))]
    y_probs = predictions_df[prob_cols].values
    
    fig = go.Figure()
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{class_name} (AUC={roc_auc:.3f})',
            line=dict(width=2)
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray', width=1)
    ))
    
    fig.update_layout(
        title=f'ROC Curves - {task_name.replace("_", " ").title()} (Test Set)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template=TEMPLATE,
        font=dict(size=12),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.6, y=0.1)
    )
    
    save_figure(fig, output_path)


def plot_precision_recall_curves(predictions_df: pd.DataFrame, task_name: str,
                                 class_names: List[str], output_path: Path):
    """Precision-Recall curves for multi-class classification."""
    logger.info(f"Creating PR curves for {task_name}...")
    
    y_true = predictions_df[f'{task_name}_true_idx'].values
    
    # Get probability columns
    prob_cols = [f'{task_name}_prob_{i}' for i in range(len(class_names))]
    y_probs = predictions_df[prob_cols].values
    
    fig = go.Figure()
    
    # Plot PR curve for each class
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
        avg_precision = average_precision_score(y_true_binary, y_score)
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{class_name} (AP={avg_precision:.3f})',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f'Precision-Recall Curves - {task_name.replace("_", " ").title()} (Test Set)',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template=TEMPLATE,
        font=dict(size=12),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.05, y=0.2)
    )
    
    save_figure(fig, output_path)


def create_performance_table(metrics: Dict, output_path: Path):
    """Performance metrics summary table."""
    logger.info("Creating performance metrics table...")
    
    data = []
    for task_name, task_metrics in metrics.items():
        data.append({
            'Task': task_name.replace('_', ' ').title(),
            'Accuracy': f"{task_metrics['accuracy']:.4f}",
            'Balanced Accuracy': f"{task_metrics['balanced_accuracy']:.4f}",
            'F1-Macro': f"{task_metrics['f1_macro']:.4f}",
            'F1-Weighted': f"{task_metrics['f1_weighted']:.4f}",
            'Precision-Macro': f"{task_metrics['precision_macro']:.4f}",
            'Recall-Macro': f"{task_metrics['recall_macro']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV and LaTeX
    csv_path = output_path.with_suffix('.csv')
    latex_path = output_path.with_suffix('.tex')
    
    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False, float_format="%.4f")
    
    logger.info(f"Saved: {csv_path}")
    logger.info(f"Saved: {latex_path}")
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#4CAF50',
            align='center',
            font=dict(color='white', size=14, family='Arial Black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[['#f0f0f0', 'white'] * len(df)],
            align='center',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title='Final Performance Metrics Summary (Test Set)',
        template=TEMPLATE,
        font=dict(size=12)
    )
    
    save_figure(fig, output_path, width=1400, height=400)


def create_per_class_table(classification_report: Dict, task_name: str,
                          output_path: Path):
    """Per-class metrics table for a task."""
    logger.info(f"Creating per-class metrics table for {task_name}...")
    
    classes = [k for k in classification_report.keys()
               if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    data = []
    for cls in classes:
        metrics = classification_report[cls]
        data.append({
            'Class': cls,
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1-score']:.4f}",
            'Support': int(metrics['support'])
        })
    
    # Add averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in classification_report:
            metrics = classification_report[avg_type]
            data.append({
                'Class': avg_type.title(),
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1-score']:.4f}",
                'Support': int(metrics['support'])
            })
    
    df = pd.DataFrame(data)
    
    # Save CSV and LaTeX
    csv_path = output_path.with_suffix('.csv')
    latex_path = output_path.with_suffix('.tex')
    
    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False)
    
    logger.info(f"Saved: {csv_path}")
    logger.info(f"Saved: {latex_path}")


def create_hyperparameters_table(config: Dict, output_path: Path):
    """Model hyperparameters table."""
    logger.info("Creating hyperparameters table...")
    
    data = []
    
    # Model architecture
    if 'hyperparameters' in config and 'model_params' in config['hyperparameters']:
        model_config = config['hyperparameters']['model_params']
        data.append({'Parameter': 'Hidden Dimensions', 'Value': str(model_config.get('hidden_dims', 'N/A'))})
        data.append({'Parameter': 'Dropout Rate', 'Value': f"{model_config.get('dropout', 0):.4f}"})
        data.append({'Parameter': 'Activation Function', 'Value': str(model_config.get('activation', 'N/A'))})
        data.append({'Parameter': 'Batch Normalization', 'Value': str(model_config.get('use_batch_norm', 'N/A'))})
    
    # Training parameters
    if 'hyperparameters' in config:
        hyperparams = config['hyperparameters']
        if 'trainer_params' in hyperparams:
            trainer = hyperparams['trainer_params']
            data.append({'Parameter': 'Learning Rate', 'Value': f"{trainer.get('learning_rate', 0):.6f}"})
            data.append({'Parameter': 'Weight Decay', 'Value': f"{trainer.get('weight_decay', 0):.6f}"})
        
        data.append({'Parameter': 'Batch Size', 'Value': str(hyperparams.get('batch_size', config.get('batch_size', 'N/A')))})
    
    data.append({'Parameter': 'Max Epochs', 'Value': str(config.get('max_epochs', 'N/A'))})
    data.append({'Parameter': 'Early Stopping Patience', 'Value': str(config.get('early_stopping_patience', 'N/A'))})
    data.append({'Parameter': 'Validation Split', 'Value': str(config.get('validation_split', 'N/A'))})
    
    # Task weights
    if 'hyperparameters' in config and 'trainer_params' in config['hyperparameters']:
        if 'task_weights' in config['hyperparameters']['trainer_params']:
            task_weights = config['hyperparameters']['trainer_params']['task_weights']
            for task, weight in task_weights.items():
                data.append({'Parameter': f'Task Weight ({task})', 'Value': f"{weight:.4f}"})
    
    df = pd.DataFrame(data)
    
    # Save CSV and LaTeX
    csv_path = output_path.with_suffix('.csv')
    latex_path = output_path.with_suffix('.tex')
    
    df.to_csv(csv_path, index=False)
    df.to_latex(latex_path, index=False)
    
    logger.info(f"Saved: {csv_path}")
    logger.info(f"Saved: {latex_path}")
    
    # Create interactive table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#4CAF50',
            align='left',
            font=dict(color='white', size=14, family='Arial Black')
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[['#f0f0f0', 'white'] * len(df)],
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title='Model Hyperparameters',
        template=TEMPLATE,
        font=dict(size=12)
    )
    
    save_figure(fig, output_path, width=1000, height=max(400, len(data) * 30 + 100))


def main():
    parser = argparse.ArgumentParser(description='Generate all paper plots and tables with Plotly')
    parser.add_argument('--metrics', type=str,
                       default='results/test_evaluation/test_metrics.json')
    parser.add_argument('--history', type=str,
                       default='results/full_training/training_history.json')
    parser.add_argument('--config', type=str,
                       default='results/full_training/final_training_config.json')
    parser.add_argument('--predictions', type=str,
                       default='results/test_evaluation/test_predictions.tsv')
    parser.add_argument('--label-encoders', type=str,
                       default='results/full_training/label_encoders.json')
    parser.add_argument('--output-dir', type=str, default='paper')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'figures' / 'model_evaluation'
    tables_dir = output_dir / 'tables' / 'model_evaluation'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Generating Paper Plots and Tables (Plotly) ===\n")
    
    # Load data
    metrics, history, config, predictions_df, encoders = load_data(
        Path(args.metrics), Path(args.history), Path(args.config),
        Path(args.predictions), Path(args.label_encoders)
    )
    
    # 1. Multi-Task Performance Summary
    plot_multitask_performance_summary(
        metrics,
        figures_dir / 'test_set_multitask_performance_summary'
    )
    
    # 2-3. Confusion Matrices and Per-Class Metrics
    for task_name, task_metrics in metrics.items():
        class_names = encoders[task_name]['classes']
        
        # Confusion Matrix
        cm = np.array(task_metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm, class_names, task_name,
            figures_dir / f'test_set_confusion_matrix_{task_name}'
        )
        
        # Per-Class Metrics Bar Chart
        plot_per_class_metrics(
            task_metrics['classification_report'],
            task_name,
            figures_dir / f'test_set_per_class_metrics_{task_name}'
        )
        
        # Per-Class Metrics Table
        create_per_class_table(
            task_metrics['classification_report'],
            task_name,
            tables_dir / f'test_set_per_class_metrics_{task_name}'
        )
    
    # 4. Training & Validation Loss
    plot_training_curves(
        history,
        figures_dir / 'training_set_loss_curves'
    )
    
    # 5-6. ROC and PR Curves
    for task_name in metrics.keys():
        class_names = encoders[task_name]['classes']
        
        plot_roc_curves(
            predictions_df, task_name, class_names,
            figures_dir / f'test_set_roc_curves_{task_name}'
        )
        
        plot_precision_recall_curves(
            predictions_df, task_name, class_names,
            figures_dir / f'test_set_pr_curves_{task_name}'
        )
    
    # 7. Performance Summary Table
    create_performance_table(
        metrics,
        tables_dir / 'test_set_performance_summary'
    )
    
    # 8. Hyperparameters Table
    create_hyperparameters_table(
        config,
        tables_dir / 'hyperparameters'
    )
    
    logger.info("\n=== All plots and tables generated successfully! ===")
    logger.info(f"Figures (HTML + PNG): {figures_dir}")
    logger.info(f"Tables (HTML + PNG + CSV + LaTeX): {tables_dir}")


if __name__ == '__main__':
    main()
