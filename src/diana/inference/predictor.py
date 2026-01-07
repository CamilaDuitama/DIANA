"""
Predictor for Diana Model

Loads trained models and performs inference on new samples.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import logging

from ..models.multitask_mlp import MultiTaskMLP
from ..models.single_task_mlp import SingleTaskMLP

logger = logging.getLogger(__name__)


class Predictor:
    """
    Load trained Diana models and predict on new samples.
    
    Supports both multi-task and single-task models.
    
    Example:
        >>> predictor = Predictor("models/multitask/best_model.pth")
        >>> features = extract_features_from_fastq("sample.fastq.gz", ...)
        >>> predictions = predictor.predict(features)
        >>> print(predictions["sample_type"])  # ancient or modern
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None
    ):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model checkpoint (.pth file).
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        model_state = checkpoint['model_state_dict']
        
        # Determine model type from checkpoint
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Infer from state dict keys
            state_keys = list(model_state.keys())
            model_type = 'multitask' if any('heads' in k for k in state_keys) else 'single_task'
        
        self.model_type = model_type
        self.config = checkpoint.get('config', {})
        
        # Reconstruct model architecture from state dict
        if model_type == 'multitask':
            # Infer input_dim from first layer
            input_dim = model_state['backbone.0.weight'].shape[1]
            
            # Check if BatchNorm was used (look for running_mean/running_var)
            use_batch_norm = any('running_mean' in k or 'running_var' in k for k in model_state.keys())
            
            # Infer hidden dims from backbone Linear layers only
            # Pattern: backbone.{idx}.weight where idx corresponds to nn.Linear layers
            # Skip BatchNorm layers (they also have .weight but are followed by .running_mean)
            hidden_dims = []
            backbone_keys = sorted([k for k in model_state.keys() if k.startswith('backbone.')])
            
            for key in backbone_keys:
                if key.endswith('.weight'):
                    # Check if this is a Linear layer (not BatchNorm)
                    # BatchNorm layers have corresponding .running_mean keys
                    base_key = key.replace('.weight', '')
                    is_batch_norm = f'{base_key}.running_mean' in model_state
                    
                    if not is_batch_norm:
                        # This is a Linear layer
                        layer_output_dim = model_state[key].shape[0]
                        hidden_dims.append(layer_output_dim)
            
            # Infer num_classes for each task from heads
            # Find the final (output) layer for each task
            num_classes = {}
            for key in model_state.keys():
                if key.startswith('heads.') and key.endswith('.weight'):
                    parts = key.split('.')
                    task_name = parts[1]
                    layer_idx = int(parts[2])
                    output_dim = model_state[key].shape[0]
                    
                    # Keep the largest layer index (final output layer)
                    if task_name not in num_classes:
                        num_classes[task_name] = {'dim': output_dim, 'idx': layer_idx}
                    elif layer_idx > num_classes[task_name]['idx']:
                        num_classes[task_name] = {'dim': output_dim, 'idx': layer_idx}
            
            # Extract just the dimensions
            num_classes = {task: info['dim'] for task, info in num_classes.items()}
            
            # Infer dropout (default to 0.5 if not in checkpoint)
            dropout = checkpoint.get('dropout', 0.5)
            
            logger.info(f"Inferred architecture: input_dim={input_dim}, hidden_dims={hidden_dims}, num_classes={num_classes}, use_batch_norm={use_batch_norm}")
            
            self.model = MultiTaskMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
        else:
            # Only multi-task models are supported
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                "Only 'multitask' models are currently supported."
            )
        
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded {model_type} model successfully")
        
    def predict(
        self,
        features: np.ndarray,
        return_probabilities: bool = False
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Predict labels for feature vector.
        
        Args:
            features: Feature vector of shape (num_unitigs,).
            return_probabilities: If True, return class probabilities.
            
        Returns:
            Dictionary with predictions for each target.
            If return_probabilities=True, includes probabilities for each class.
        """
        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.model(x)
            
            predictions = {}
            for target, logits in outputs.items():
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = int(np.argmax(probs))
                
                if return_probabilities:
                    predictions[target] = {
                        'class': pred_class,
                        'probabilities': probs.tolist()
                    }
                else:
                    predictions[target] = pred_class
            
            return predictions
    
    def explain_prediction(
        self,
        features: np.ndarray,
        top_k: int = 20
    ) -> Dict[str, Dict]:
        """
        Explain which features contributed most to the prediction.
        
        Uses gradient-based attribution to identify the most important
        features (unitigs) that influenced the prediction for each task.
        
        Args:
            features: Feature vector of shape (num_unitigs,).
            top_k: Number of top contributing features to return per task.
            
        Returns:
            Dictionary with feature importance for each task:
            {
                'sample_type': {
                    'predicted_class': 0,
                    'predicted_label': 'ancient_metagenome',
                    'top_features': [
                        {'index': 123, 'importance': 0.456, 'value': 1.0},
                        {'index': 45, 'importance': 0.389, 'value': 1.0},
                        ...
                    ]
                },
                ...
            }
        
        Example:
            >>> predictor = Predictor('model.pth')
            >>> features = extract_features('sample.fastq')
            >>> explanation = predictor.explain_prediction(features, top_k=10)
            >>> print(f"Predicted: {explanation['sample_type']['predicted_label']}")
            >>> print("Top distinguishing unitigs:")
            >>> for feat in explanation['sample_type']['top_features'][:5]:
            >>>     print(f"  Unitig {feat['index']}: importance={feat['importance']:.3f}")
        """
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        x.requires_grad = True
        
        # Forward pass
        self.model.eval()
        outputs = self.model(x)
        
        explanations = {}
        
        for target, logits in outputs.items():
            # Get predicted class
            probs = torch.softmax(logits, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())
            
            # Compute gradient of predicted class logit w.r.t. input
            if x.grad is not None:
                x.grad.zero_()
            
            logits[0, pred_class].backward(retain_graph=True)
            
            # Feature importance = absolute gradient * feature value
            # This shows which features, when present, most influenced this prediction
            gradients = x.grad.abs().squeeze(0).cpu().numpy()
            feature_importance = gradients * features
            
            # Get top-k features
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            
            top_features = []
            for idx in top_indices:
                top_features.append({
                    'index': int(idx),
                    'importance': float(feature_importance[idx]),
                    'gradient': float(gradients[idx]),
                    'value': float(features[idx])
                })
            
            explanations[target] = {
                'predicted_class': pred_class,
                'confidence': float(probs[0, pred_class].item()),
                'top_features': top_features
            }
        
        return explanations
    
    def predict_batch(
        self,
        features_batch: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """
        Predict labels for batch of feature vectors.
        
        Args:
            features_batch: Array of shape (num_samples, num_unitigs).
            batch_size: Batch size for inference.
            
        Returns:
            Dictionary with predictions for each target as arrays.
        """
        raise NotImplementedError("Batch prediction not yet implemented")
