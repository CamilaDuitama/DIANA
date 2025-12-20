"""
Configuration Manager for DIANA
================================

Handles loading, merging, validation, and saving of configuration files.

Features:
  - Load from YAML files
  - Merge with defaults
  - Validate required fields
  - Support for environment variable substitution
  - Save configurations for reproducibility
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

from .defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for DIANA experiments.
    
    Supports:
      - Loading from YAML/JSON files
      - Merging with default configuration
      - Environment variable substitution (${VAR_NAME})
      - Validation of required fields
      - Saving for reproducibility
    
    Example:
        # Load config from file
        config = ConfigManager.from_yaml("configs/experiment1.yaml")
        
        # Override specific values
        config.set("training.max_epochs", 100)
        
        # Get values
        epochs = config.get("training.max_epochs")
        
        # Save for reproducibility
        config.save("results/experiment1/config_used.yaml")
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        if config_dict is None:
            self.config = deepcopy(DEFAULT_CONFIG)
        else:
            # Merge with defaults (user config overrides defaults)
            self.config = self._deep_merge(deepcopy(DEFAULT_CONFIG), config_dict)
        
        # Substitute environment variables
        self.config = self._substitute_env_vars(self.config)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'ConfigManager':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ConfigManager instance
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        logger.info(f"Loading configuration from {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        return cls(user_config)
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'ConfigManager':
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        logger.info(f"Loading configuration from {json_path}")
        
        with open(json_path, 'r') as f:
            user_config = json.load(f)
        
        return cls(user_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "training.max_epochs")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "training.max_epochs")
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: Path, format: str = 'yaml'):
        """
        Save configuration to file.
        
        Args:
            save_path: Path to save configuration
            format: 'yaml' or 'json'
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'yaml':
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {save_path}")
    
    def validate(self, required_keys: list = None):
        """
        Validate configuration has required keys.
        
        Args:
            required_keys: List of required keys (dot notation)
            
        Raises:
            ValueError: If required keys are missing
        """
        if required_keys is None:
            required_keys = [
                "data.train_matrix",
                "data.train_metadata",
                "output.base_dir",
                "training.max_epochs",
            ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")
        
        logger.info("Configuration validation passed")
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge override into base."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in config."""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            var_name = config[2:-1]
            return os.environ.get(var_name, config)
        else:
            return config
    
    def to_dict(self) -> dict:
        """Return configuration as dictionary."""
        return deepcopy(self.config)
    
    def __repr__(self) -> str:
        return f"ConfigManager({yaml.dump(self.config, default_flow_style=False)})"
