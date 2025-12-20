"""
Unit tests for configuration management.

Tests ConfigManager for loading, merging, and validating configuration files.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from diana.config.manager import ConfigManager


class TestConfigManager:
    """Test the ConfigManager class for handling YAML/JSON configs."""
    
    def test_default_config(self):
        """Verify default configuration has required structure."""
        config = ConfigManager()
        
        required_keys = ['data', 'output', 'training', 'model', 'optimizer', 'optuna', 'logging']
        for key in required_keys:
            assert config.get(key) is not None, f"Missing key: {key}"
    
    def test_nested_access(self):
        """Test accessing nested config values with dot notation."""
        config = ConfigManager()
        
        max_epochs = config.get('training.max_epochs')
        assert max_epochs is not None
        assert isinstance(max_epochs, int)
        assert max_epochs > 0
    
    def test_default_fallback(self):
        """Test default value when key doesn't exist."""
        config = ConfigManager()
        
        value = config.get('nonexistent.key', default='fallback')
        assert value == 'fallback'
    
    def test_load_yaml(self, dummy_config_path):
        """Test loading configuration from YAML file."""
        config = ConfigManager.from_yaml(str(dummy_config_path))
        
        assert config.get('training.max_epochs') == 2
        assert config.get('optimizer.batch_size') == 4
    
    def test_yaml_overrides_defaults(self, dummy_config_path):
        """Verify YAML values override defaults."""
        config = ConfigManager.from_yaml(str(dummy_config_path))
        
        # Test YAML specifies 2 epochs
        assert config.get('training.max_epochs') == 2
    
    def test_set_value(self):
        """Test programmatically setting config values."""
        config = ConfigManager()
        
        config.set('custom.test', 42)
        assert config.get('custom.test') == 42
    
    def test_save_yaml(self):
        """Test saving configuration to YAML file."""
        config = ConfigManager()
        config.set('test.value', 123)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path, format='yaml')
            
            with open(temp_path) as f:
                loaded = yaml.safe_load(f)
            
            assert 'test' in loaded
            assert loaded['test']['value'] == 123
        finally:
            Path(temp_path).unlink()
    
    def test_save_json(self):
        """Test saving configuration to JSON file."""
        config = ConfigManager()
        config.set('test.value', 456)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save(temp_path, format='json')
            
            with open(temp_path) as f:
                loaded = json.load(f)
            
            assert loaded['test']['value'] == 456
        finally:
            Path(temp_path).unlink()
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ConfigManager()
        config.set('test.nested.value', 42)
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['test']['nested']['value'] == 42


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_required_keys_present(self):
        """Verify all required configuration keys exist."""
        config = ConfigManager()
        
        required = [
            'training.max_epochs',
            'optimizer.learning_rate',
            'model.hidden_dims',
        ]
        
        for key in required:
            assert config.get(key) is not None, f"Missing required key: {key}"
    
    def test_numeric_values_valid(self):
        """Verify numeric config values are positive."""
        config = ConfigManager()
        
        lr = config.get('optimizer.learning_rate')
        assert lr > 0
        
        epochs = config.get('training.max_epochs')
        assert epochs > 0
