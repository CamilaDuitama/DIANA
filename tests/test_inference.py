"""
Tests for inference pipeline components.

Validates that the prediction pipeline works correctly end-to-end,
including model loading, feature extraction detection, and output formatting.
"""

import pytest
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from diana.inference.predictor import Predictor
from diana.models.multitask_mlp import MultiTaskMLP


class TestPredictor:
    """Test the Predictor class for loading models and making predictions."""
    
    def test_predictor_loads_checkpoint(self, temp_dir):
        """Verify Predictor correctly loads saved model checkpoints."""
        # Create a minimal checkpoint
        input_dim = 100
        num_classes = {
            "sample_type": 2,
            "community_type": 6,
            "sample_host": 12,
            "material": 13
        }
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            num_classes=num_classes,
            dropout=0.3
        )
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_model.pth"
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim},
            'val_loss': 0.5,
            'history': {'loss': [1.0, 0.8, 0.6]}
        }, checkpoint_path)
        
        # Load with Predictor
        predictor = Predictor(checkpoint_path, device='cpu')
        
        assert predictor.model_type == 'multitask'
        assert predictor.model is not None
        assert predictor.device == 'cpu'
    
    def test_predictor_inference_shapes(self, temp_dir):
        """Verify predictions have correct shape and structure."""
        # Create and save model
        input_dim = 50
        num_classes = {
            "sample_type": 2,
            "community_type": 6,
            "sample_host": 12,
            "material": 13
        }
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[32, 16],
            num_classes=num_classes
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, checkpoint_path)
        
        # Load and predict
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Create dummy features
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Check structure
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == {'sample_type', 'community_type', 'sample_host', 'material'}
        
        # Check each task has predictions and probabilities
        for task, pred in predictions.items():
            assert 'class' in pred
            assert 'probabilities' in pred
            
            # Probabilities should sum to ~1.0
            prob_sum = sum(pred['probabilities'])
            assert 0.99 <= prob_sum <= 1.01, f"{task} probabilities sum to {prob_sum}"
    
    def test_prediction_output_format(self, temp_dir):
        """Verify prediction JSON has all required fields and valid values."""
        # Setup model
        input_dim = 20
        num_classes = {"sample_type": 2, "community_type": 3}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Validate predicted class is valid integer
        for task, pred in predictions.items():
            assert isinstance(pred['class'], int)
            assert pred['class'] >= 0
            
            # Probabilities should be non-negative and sum to ~1
            assert all(p >= 0.0 for p in pred['probabilities']), \
                f"{task} has negative probabilities"
            prob_sum = sum(pred['probabilities'])
            assert 0.99 <= prob_sum <= 1.01


class TestPairedEndDetection:
    """Test detection of paired-end FASTQ files."""
    
    def test_paired_end_patterns(self, temp_dir):
        """Test detection of various R1/R2 naming patterns."""
        from diana.cli.predict import detect_paired_end
        
        # Test case 1: _1 / _2 pattern
        r1_file = temp_dir / "sample_1.fastq.gz"
        r2_file = temp_dir / "sample_2.fastq.gz"
        r1_file.touch()
        r2_file.touch()
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 2
        assert r1_file in detected
        assert r2_file in detected
        
        # Test case 2: _R1 / _R2 pattern
        r1_file = temp_dir / "ERR123_R1.fastq.gz"
        r2_file = temp_dir / "ERR123_R2.fastq.gz"
        r1_file.touch()
        r2_file.touch()
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 2
        assert r1_file in detected
        assert r2_file in detected
    
    def test_single_end_detection(self, temp_dir):
        """Single-end files should return list of length 1."""
        from diana.cli.predict import detect_paired_end
        
        single_file = temp_dir / "sample.fastq.gz"
        single_file.touch()
        
        detected = detect_paired_end(single_file)
        assert len(detected) == 1
        assert detected[0] == single_file
    
    def test_missing_pair_returns_single(self, temp_dir):
        """If R2 doesn't exist, should return only R1."""
        from diana.cli.predict import detect_paired_end
        
        r1_file = temp_dir / "sample_R1.fastq.gz"
        r1_file.touch()
        # R2 does not exist
        
        detected = detect_paired_end(r1_file)
        assert len(detected) == 1
        assert detected[0] == r1_file


class TestModelArchitectureReconstruction:
    """Test that model architecture is correctly inferred from checkpoints."""
    
    def test_infer_batch_norm_from_checkpoint(self, temp_dir):
        """Verify batch norm detection from checkpoint keys."""
        # Model WITH batch norm
        model_bn = MultiTaskMLP(
            input_dim=50,
            hidden_dims=[32],
            num_classes={"task1": 2},
            use_batch_norm=True
        )
        
        checkpoint_path = temp_dir / "model_bn.pth"
        torch.save({
            'model_state_dict': model_bn.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Should successfully load (batch norm detected)
        assert predictor.model is not None
        
        # Model WITHOUT batch norm
        model_no_bn = MultiTaskMLP(
            input_dim=50,
            hidden_dims=[32],
            num_classes={"task1": 2},
            use_batch_norm=False
        )
        
        checkpoint_path2 = temp_dir / "model_no_bn.pth"
        torch.save({
            'model_state_dict': model_no_bn.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path2)
        
        predictor2 = Predictor(checkpoint_path2, device='cpu')
        assert predictor2.model is not None
    
    def test_infer_hidden_dims_from_checkpoint(self, temp_dir):
        """Verify hidden dimensions are correctly inferred."""
        hidden_dims = [128, 64, 32]
        
        model = MultiTaskMLP(
            input_dim=100,
            hidden_dims=hidden_dims,
            num_classes={"task1": 3}
        )
        
        checkpoint_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, checkpoint_path)
        
        predictor = Predictor(checkpoint_path, device='cpu')
        
        # Model should load successfully
        assert predictor.model is not None
        
        # Test inference works
        features = np.random.rand(100).astype(np.float32)
        predictions = predictor.predict(features)
        assert 'task1' in predictions


class TestLabelDecoding:
    """Test label encoding/decoding for predictions."""
    
    def test_label_encoder_creation_and_usage(self, temp_dir):
        """Test creating and using sklearn label encoders."""
        labels = ['ancient', 'modern', 'environmental']
        encoder = LabelEncoder()
        encoder.fit(labels)
        
        # Test encoding
        encoded = encoder.transform(['ancient', 'modern'])
        assert len(encoded) == 2
        assert encoded[0] != encoded[1]
        
        # Test decoding
        decoded = encoder.inverse_transform(encoded)
        assert list(decoded) == ['ancient', 'modern']
        
        # Test classes
        assert list(encoder.classes_) == sorted(labels)
    
    def test_label_encoder_json_format(self, temp_dir):
        """Test saving/loading label encoders in JSON format (project convention)."""
        # Create encoders
        encoders = {
            'sample_type': LabelEncoder(),
            'material': LabelEncoder()
        }
        encoders['sample_type'].fit(['ancient', 'modern', 'environmental'])
        encoders['material'].fit(['bone', 'soil', 'sediment', 'calculus'])
        
        # Save in JSON format (as done in training)
        encoders_json = {}
        for task_name, encoder in encoders.items():
            encoders_json[task_name] = {
                'classes': list(encoder.classes_)
            }
        
        json_path = temp_dir / "label_encoders.json"
        with open(json_path, 'w') as f:
            json.dump(encoders_json, f, indent=2)
        
        # Load and reconstruct
        with open(json_path, 'r') as f:
            loaded_json = json.load(f)
        
        loaded_encoders = {}
        for task_name, encoder_data in loaded_json.items():
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_data['classes'])
            loaded_encoders[task_name] = encoder
        
        # Test loaded encoders work
        for task_name in encoders.keys():
            original_classes = list(encoders[task_name].classes_)
            loaded_classes = list(loaded_encoders[task_name].classes_)
            assert original_classes == loaded_classes
    
    def test_predictions_with_label_decoding(self, temp_dir):
        """Test full prediction workflow with label decoding."""
        # Create model and label encoders
        input_dim = 50
        num_classes = {"sample_type": 3, "material": 4}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[32],
            num_classes=num_classes
        )
        
        # Create label encoders
        encoders = {
            'sample_type': LabelEncoder(),
            'material': LabelEncoder()
        }
        encoders['sample_type'].fit(['ancient', 'modern', 'environmental'])
        encoders['material'].fit(['bone', 'soil', 'sediment', 'calculus'])
        
        # Save model
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        # Save encoders in JSON format
        encoders_json = {}
        for task_name, encoder in encoders.items():
            encoders_json[task_name] = {'classes': list(encoder.classes_)}
        
        with open(temp_dir / "label_encoders.json", 'w') as f:
            json.dump(encoders_json, f)
        
        # Make predictions
        predictor = Predictor(model_path, device='cpu')
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Decode predictions
        for task_name in num_classes.keys():
            encoder = encoders[task_name]
            pred_class_idx = predictions[task_name]['class']
            decoded_label = encoder.inverse_transform([pred_class_idx])[0]
            
            # Should be one of the valid labels
            assert decoded_label in encoder.classes_


class TestPredictionOutputFormat:
    """Test the structure and validity of prediction outputs."""
    
    def test_json_output_structure(self, temp_dir):
        """Test that predictions can be serialized to JSON with correct structure."""
        # Create simple model
        input_dim = 30
        num_classes = {
            "sample_type": 2,
            "community_type": 3,
            "sample_host": 4,
            "material": 5
        }
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        # Make predictions
        predictor = Predictor(model_path, device='cpu')
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Convert to JSON
        json_path = temp_dir / "predictions.json"
        with open(json_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Load and validate
        with open(json_path, 'r') as f:
            loaded_predictions = json.load(f)
        
        # Check all tasks present
        assert set(loaded_predictions.keys()) == set(num_classes.keys())
        
        # Check each task has required fields
        for task_name, pred in loaded_predictions.items():
            assert 'class' in pred
            assert 'probabilities' in pred
            assert isinstance(pred['class'], int)
            assert isinstance(pred['probabilities'], list)
            assert len(pred['probabilities']) == num_classes[task_name]
            
            # All probabilities should be non-negative floats
            assert all(isinstance(p, float) for p in pred['probabilities'])
            assert all(p >= 0.0 for p in pred['probabilities'])
            
            # Sum should be close to 1.0
            prob_sum = sum(pred['probabilities'])
            assert 0.99 <= prob_sum <= 1.01
    
    def test_batch_predictions(self, temp_dir):
        """Test making predictions on multiple samples."""
        input_dim = 25
        num_classes = {"task1": 3}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        predictor = Predictor(model_path, device='cpu')
        
        # Create batch of features
        n_samples = 5
        batch_features = [
            np.random.rand(input_dim).astype(np.float32)
            for _ in range(n_samples)
        ]
        
        # Make predictions for each sample
        batch_predictions = []
        for features in batch_features:
            preds = predictor.predict(features, return_probabilities=True)
            batch_predictions.append(preds)
        
        # All predictions should have same structure
        assert len(batch_predictions) == n_samples
        for preds in batch_predictions:
            assert 'task1' in preds
            assert 'class' in preds['task1']
            assert 'probabilities' in preds['task1']


class TestFeatureValidation:
    """Test validation of input features."""
    
    def test_wrong_feature_dimensions_raises_error(self, temp_dir):
        """Test that features with wrong dimensions raise clear error."""
        input_dim = 100
        num_classes = {"task1": 2}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[32],
            num_classes=num_classes
        )
        
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        predictor = Predictor(model_path, device='cpu')
        
        # Try to predict with wrong dimensions
        wrong_features = np.random.rand(50).astype(np.float32)  # Should be 100
        
        with pytest.raises((RuntimeError, ValueError)):
            predictor.predict(wrong_features)
    
    def test_nan_features_handling(self, temp_dir):
        """Test handling of NaN values in features."""
        input_dim = 30
        num_classes = {"task1": 2}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        predictor = Predictor(model_path, device='cpu')
        
        # Create features with NaN
        features = np.random.rand(input_dim).astype(np.float32)
        features[5] = np.nan
        
        # Should either raise an error or handle gracefully
        # (depending on implementation)
        try:
            predictions = predictor.predict(features)
            # If it doesn't raise, predictions should still be valid
            assert 'task1' in predictions
        except (ValueError, RuntimeError):
            # Expected behavior - rejecting invalid input
            pass
    
    def test_zero_features(self, temp_dir):
        """Test prediction on all-zero features."""
        input_dim = 20
        num_classes = {"task1": 2}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[16],
            num_classes=num_classes
        )
        
        model_path = temp_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {'input_dim': input_dim}
        }, model_path)
        
        predictor = Predictor(model_path, device='cpu')
        
        # All zeros (empty sample)
        features = np.zeros(input_dim, dtype=np.float32)
        predictions = predictor.predict(features, return_probabilities=True)
        
        # Should still make a prediction
        assert 'task1' in predictions
        assert 'probabilities' in predictions['task1']
        
        # Probabilities should still sum to 1
        prob_sum = sum(predictions['task1']['probabilities'])
        assert 0.99 <= prob_sum <= 1.01


class TestModelCheckpointCompatibility:
    """Test loading models from different checkpoint formats."""
    
    def test_load_checkpoint_with_history(self, temp_dir):
        """Test loading checkpoint that includes training history."""
        input_dim = 40
        num_classes = {"task1": 2}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[24],
            num_classes=num_classes
        )
        
        # Save with extensive metadata
        model_path = temp_dir / "model.pth"
        torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask',
            'config': {
                'input_dim': input_dim,
                'hidden_dims': [24],
                'dropout': 0.3
            },
            'val_loss': 0.234,
            'val_accuracy': 0.876,
            'history': {
                'loss': [1.0, 0.8, 0.6, 0.4, 0.234],
                'accuracy': [0.5, 0.6, 0.7, 0.8, 0.876]
            },
            'optimizer_state_dict': None  # Not needed for inference
        }, model_path)
        
        # Should load successfully
        predictor = Predictor(model_path, device='cpu')
        assert predictor.model is not None
        
        # Should be able to make predictions
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features)
        assert 'task1' in predictions
    
    def test_load_checkpoint_minimal(self, temp_dir):
        """Test loading minimal checkpoint (just state dict)."""
        input_dim = 30
        num_classes = {"task1": 3}
        
        model = MultiTaskMLP(
            input_dim=input_dim,
            hidden_dims=[20],
            num_classes=num_classes
        )
        
        # Save minimal checkpoint
        model_path = temp_dir / "model_minimal.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'multitask'
        }, model_path)
        
        # Should still load
        predictor = Predictor(model_path, device='cpu')
        assert predictor.model is not None
        
        features = np.random.rand(input_dim).astype(np.float32)
        predictions = predictor.predict(features)
        assert 'task1' in predictions

