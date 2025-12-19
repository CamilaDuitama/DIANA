"""
Configuration loading and saving utilities.

Handles YAML and JSON configuration files used throughout the project.
All scripts load their configuration via load_config().
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_file: Path = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def save_config(config: Dict[str, Any], save_path: Path):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration (.yaml, .yml, or .json)
        
    Raises:
        ValueError: If file format is not supported
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        if save_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {save_path.suffix}")
