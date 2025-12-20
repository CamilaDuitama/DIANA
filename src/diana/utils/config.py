"""
Configuration and Logging Utilities
====================================

Handles configuration loading/saving and logging setup for DIANA.

DEPENDENCIES:
-------------
Python packages:
  - yaml (PyYAML)
  - json (standard library)
  - logging (standard library)

Features:
  - YAML/JSON configuration loading
  - Structured logging to file + console
  - Log rotation support
  - Configurable log levels
"""

import yaml
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logging(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5
):
    """
    Setup logging configuration with file rotation and console output.
    
    Args:
        log_file: Path to log file (auto-generated if None)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_to_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = Path(f"logs/diana_{timestamp}.log")
        
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


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
