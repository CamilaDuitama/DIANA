#!/usr/bin/env python3
"""
Compute balanced accuracy and F1 macro for SEEN validation labels only
"""

import json
import numpy as np
from pathlib import Path


def parse_confusion_matrix(cm_dict):
    """Parse confusion matrix from dict format to sklearn format"""
    all_classes = set()
    for key in cm_dict.keys():
        true_class, pred_class = key.split('_to_')
        all_classes.add(true_class)
        all_classes.add(pred_class)
    
    classes = sorted(list(all_classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for key, count in cm_dict.items():
        true_class, pred_class = key.split('_to_')
        true_idx = class_to_idx[true_class]
        pred_idx = class_to_idx[pred_class]
        cm[true_idx, pred_idx] = count
    
    return cm, classes


def compute_balanced_accuracy(cm):
    """Compute balanced accuracy from confusion matrix"""
    per_class_recall = []
    for i in range(cm.shape[0]):
        class_total = cm[i, :].sum()
        if class_total > 0:
            recall = cm[i, i] / class_total
            per_class_recall.append(recall)
    
    return np.mean(per_class_recall) if per_class_recall else 0.0


def compute_precision_macro(cm):
    """Compute macro precision from confusion matrix"""
    n_classes = cm.shape[0]
    precisions = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def compute_recall_macro(cm):
    """Compute macro recall from confusion matrix"""
    n_classes = cm.shape[0]
    recalls = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def compute_f1_macro(cm):
    """Compute macro F1 from confusion matrix"""
    n_classes = cm.shape[0]
    f1_scores = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    return np.mean(f1_scores) if f1_scores else 0.0


def main():
    # Load validation comparison
    val_file = Path("paper/tables/validation/validation_comparison.json")
    with open(val_file) as f:
        val_data = json.load(f)
    
    results = {}
    
    for task_name, task_data in val_data.items():
        # Use SEEN labels only for fair comparison
        cm_dict = task_data['seen']['confusion_matrix']
        accuracy = task_data['seen']['accuracy']
        total = task_data['seen']['total']
        
        if not cm_dict or total == 0:
            print(f"⚠️  {task_name}: No seen labels (all validation labels unseen during training)")
            results[task_name] = {
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'f1_macro': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'n_classes': 0,
                'total_samples': 0,
                'note': 'No seen labels'
            }
            continue
        
        # Parse confusion matrix
        cm, classes = parse_confusion_matrix(cm_dict)
        
        # Compute metrics
        balanced_acc = compute_balanced_accuracy(cm)
        precision_macro = compute_precision_macro(cm)
        recall_macro = compute_recall_macro(cm)
        f1_macro = compute_f1_macro(cm)
        
        results[task_name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'n_classes': len(classes),
            'total_samples': total
        }
        
        print(f"\n{task_name} (SEEN labels only):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  Precision Macro: {precision_macro:.4f}")
        print(f"  Recall Macro: {recall_macro:.4f}")
        print(f"  Classes: {len(classes)}, Samples: {total}")
    
    # Save results
    output_file = Path("results/validation_predictions/validation_metrics_seen_only.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved SEEN-only metrics to {output_file}")


if __name__ == "__main__":
    main()
