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
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Determine model type from checkpoint
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
            # Infer from architecture
            model_type = 'multitask' if 'heads' in checkpoint['model_state_dict'] else 'single_task'
        
        self.model_type = model_type
        self.config = checkpoint.get('config', {})
        
        # Reconstruct model
        if model_type == 'multitask':
            self.model = MultiTaskMLP(
                input_dim=checkpoint['input_dim'],
                hidden_dims=checkpoint.get('hidden_dims', [512, 256, 128]),
                num_classes=checkpoint['num_classes'],
                dropout=checkpoint.get('dropout', 0.5)
            )
        else:
            raise NotImplementedError("Single-task model loading not yet implemented")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
