"""
Predictor for Diana Model

Loads trained models and performs inference on new samples.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import logging

from ..models.multitask_mlp import MultiTaskMLP, split_tasks
from ..models.single_task_mlp import SingleTaskMLP
from ..training.active_classes import masked_softmax

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
            model_path: Path to saved model checkpoint (.pth file), or a released-model
                folder (seed*/model.pt networks, optional lr/<task>/model_*.joblib).
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Released format only (model.pt): networks averaged, per-task class names,
        # and logistic regressions blended in by class name.
        self.networks = []
        self.class_names = None
        self.logreg = {}
        self.confidence = None   # confidence.json of a released folder: class prior, tau and no-call cutoffs

        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")

        if self.model_path.is_dir():
            self._load_released(sorted(self.model_path.rglob('seed*/model.pt')))
        else:
            self._load_model()

    def _load_released(self, paths: list, checkpoints: Optional[list] = None):
        """Load released networks (model.pt: state_dict, model_params, classes, active_classes)
        and, for a model folder, its lr/<task>/model_*.joblib logistic regressions."""
        if not paths:
            raise FileNotFoundError(f"No seed*/model.pt networks under {self.model_path}")
        checkpoints = checkpoints or [torch.load(p, map_location=self.device, weights_only=False)
                                      for p in paths]
        first = checkpoints[0]
        for p, ck in zip(paths, checkpoints):
            if (ck['task_names'], ck['classes'], ck['active_classes'], ck['input_dim']) != (
                    first['task_names'], first['classes'], first['active_classes'], first['input_dim']):
                raise ValueError(f"{p}: tasks, classes or input width differ from {paths[0]}")
            net = MultiTaskMLP(input_dim=ck['input_dim'], num_classes=ck['num_classes'], **ck['model_params'])
            net.load_state_dict(ck['state_dict'])
            self.networks.append(net.to(self.device).eval())
        self.task_names = list(first['task_names'])
        self.input_dim = int(first['input_dim'])
        self.network_classes = {t: list(first['classes'][t]) for t in self.task_names}
        self.active = {t: np.array([c in first['active_classes'][t] for c in first['classes'][t]])
                       for t in self.task_names}
        self.class_names = dict(self.network_classes)
        if self.model_path.is_dir():
            import joblib
            for p in sorted(self.model_path.glob('lr/*/model_*.joblib')):
                task = p.parent.name
                if task not in self.task_names or task in self.logreg:
                    raise ValueError(f"{p}: unknown task or a second logistic regression for {task}")
                self.logreg[task] = joblib.load(p)
                self.class_names[task] = sorted(set(self.class_names[task]) | set(map(str, self.logreg[task].classes_)))
            conf = self.model_path / 'confidence.json'
            if conf.exists():
                with open(conf) as f:
                    self.confidence = json.load(f)
        self.model_type = 'released'
        logger.info(f"Loaded {len(self.networks)} network(s){' + logistic regression for ' + ', '.join(self.logreg) if self.logreg else ''}")

    def _load_model(self):
        """Load model from checkpoint.

        Supports two checkpoint formats:
        - Dict with 'model_state_dict' key (standard DIANA training output)
        - Raw state_dict (bare OrderedDict of tensors)
        """
        # weights_only=False is required because checkpoints may contain non-tensor
        # objects (config dicts, etc.).  Only load checkpoints from trusted sources.
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Support both wrapped checkpoints and raw state dicts
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and 'model_params' in checkpoint:
            return self._load_released([self.model_path], [checkpoint])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            # Raw state dict — no metadata available
            model_state = checkpoint
            checkpoint = {}
            logger.warning(
                "Checkpoint is a raw state_dict with no metadata. "
                "Architecture will be inferred from tensor shapes."
            )
        else:
            raise ValueError(
                f"Unrecognised checkpoint format in {self.model_path}. "
                "Expected a dict with 'model_state_dict' or a raw state_dict."
            )

        # Determine model type from checkpoint or by inspecting state keys
        if 'model_type' in checkpoint:
            model_type = checkpoint['model_type']
        else:
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
            task_info = {task: info['dim'] for task, info in num_classes.items()}

            # Split classification from regression. A sibling final_training_config.json
            # is authoritative; failing that, a single-output head is a regression head
            # (MultiTaskMLP forbids single-class classification, so this is unambiguous).
            # Getting this wrong silently drops the sigmoid and pins every regression
            # prediction to 1.0.
            task_types = self.config.get('task_types')
            if task_types is None:
                cfg_path = self.model_path.parent / 'final_training_config.json'
                if cfg_path.exists():
                    with open(cfg_path) as fh:
                        task_types = json.load(fh).get('task_types')
                    if task_types:
                        logger.info(f"Loaded task_types from {cfg_path}")
            num_classes, regression_tasks = split_tasks(task_info, task_types)
            if regression_tasks:
                logger.info(f"Regression heads: {regression_tasks}")
            
            # Infer dropout (default to 0.5 if not in checkpoint)
            dropout = checkpoint.get('dropout', 0.5)
            
            logger.info(f"Inferred architecture: input_dim={input_dim}, hidden_dims={hidden_dims}, num_classes={num_classes}, use_batch_norm={use_batch_norm}")
            
            self.model = MultiTaskMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                regression_tasks=regression_tasks,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
        else:
            # Single-task model
            input_dim = model_state[list(model_state.keys())[0]].shape[1]
            use_batch_norm = any(
                'running_mean' in k or 'running_var' in k for k in model_state.keys()
            )
            hidden_dims = []
            for key in sorted(model_state.keys()):
                if key.endswith('.weight'):
                    base_key = key.replace('.weight', '')
                    if f'{base_key}.running_mean' not in model_state:
                        hidden_dims.append(model_state[key].shape[0])

            # The last entry in hidden_dims is the output (num_classes), not hidden
            num_classes = hidden_dims.pop() if hidden_dims else 2
            dropout = checkpoint.get('dropout', 0.5)

            logger.info(
                f"Inferred single-task architecture: input_dim={input_dim}, "
                f"hidden_dims={hidden_dims}, num_classes={num_classes}, "
                f"use_batch_norm={use_batch_norm}"
            )

            self.model = SingleTaskMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm
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
        if self.networks:
            predictions = {}
            for target, probs in self.predict_proba(features).items():
                pred_class = int(np.argmax(probs[0]))
                predictions[target] = ({'class': pred_class, 'probabilities': probs[0].tolist()}
                                       if return_probabilities else pred_class)
            return predictions

        with torch.no_grad():
            # Convert to tensor
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs = self.model(x)
            
            regression_tasks = set(getattr(self.model, 'regression_tasks', []) or [])

            predictions = {}
            for target, logits in outputs.items():
                if target in regression_tasks:
                    # Regression head: already sigmoid-bounded to [0, 1], shape (batch,).
                    # Still in normalised units -- the caller denormalises with the
                    # bounds in label_encoders.json.
                    predictions[target] = {'value': float(logits.cpu().numpy().reshape(-1)[0])}
                    continue

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

    def predict_proba(self, features: np.ndarray, corrected: bool = False) -> Dict[str, np.ndarray]:
        """Released model: per task, (n_samples, n_classes) probabilities in class_names order.

        The networks' softmax over their active classes, averaged; on a task with a logistic
        regression, 0.5 x that average + 0.5 x its probabilities, classes matched by name
        (a class absent from one side counts 0 there). With corrected=True each component first gets the
        training class frequencies back (P * pi^tau, renormalised; tau_network for the networks,
        tau_logistic_regression for the logistic regression; confidence.json): the probabilities behind the
        reported confidence. Labels and flag scores use corrected=False.
        """
        if corrected and self.confidence is None:
            raise ValueError("corrected probabilities need confidence.json in the released model folder")
        if not self.networks:
            raise ValueError("predict_proba needs a released model (model.pt or a folder of them)")
        X = np.atleast_2d(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            outputs = [net(torch.from_numpy(X).to(self.device)) for net in self.networks]
        probs = {}
        for t in self.task_names:
            p_net = np.mean([masked_softmax(o[t], self.active[t]).cpu().numpy() for o in outputs], axis=0)
            if corrected:
                p_net = self._add_prior(p_net, t, self.network_classes[t], self.confidence['tau_network'])
            if t not in self.logreg:
                probs[t] = p_net
                continue
            lr, names = self.logreg[t], self.class_names[t]
            p_lr = lr.predict_proba(X)
            if corrected:
                p_lr = self._add_prior(p_lr, t, [str(c) for c in lr.classes_], self.confidence['tau_logistic_regression'])
            p = np.zeros((X.shape[0], len(names)))
            p[:, [names.index(c) for c in self.network_classes[t]]] += 0.5 * p_net
            p[:, [names.index(str(c)) for c in lr.classes_]] += 0.5 * p_lr
            probs[t] = p
        return probs

    def _add_prior(self, p: np.ndarray, task: str, classes: list, tau: float) -> np.ndarray:
        """P * pi^tau over `classes` (pi: training class shares from confidence.json, 0 if absent), renormalised."""
        prior = self.confidence['class_prior'][task]
        w = np.array([prior.get(c, 0.0) for c in classes]) ** tau
        x = p * w[None, :]
        s = x.sum(axis=1, keepdims=True)
        return np.where(s > 0, x / np.where(s > 0, s, 1), p)

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
