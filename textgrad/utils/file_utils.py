"""
File utilities for TextGrad.
Provides safe file operations with automatic directory creation.
"""

import os
import json
import logging
from typing import Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_open(file_path: Union[str, Path], mode: str = "r", **kwargs):
    """
    Safely open a file, creating parent directories if needed for write modes.
    
    Args:
        file_path: Path to the file
        mode: File open mode (r, w, a, etc.)
        **kwargs: Additional arguments for open()
    
    Returns:
        File object
    """
    file_path = Path(file_path)
    
    # Create parent directories for write/append modes
    if any(m in mode for m in ['w', 'a', 'x']):
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    return open(file_path, mode, **kwargs)


def safe_save_json(data: Any, file_path: Union[str, Path], **kwargs):
    """
    Safely save data as JSON, creating parent directories if needed.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        **kwargs: Additional arguments for json.dump()
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, **kwargs)


def safe_save_text(text: str, file_path: Union[str, Path], mode: str = "w", **kwargs):
    """
    Safely save text to a file, creating parent directories if needed.
    
    Args:
        text: Text to save
        file_path: Path to save the file
        mode: File mode (w, a, etc.)
        **kwargs: Additional arguments for open()
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, mode, **kwargs) as f:
        f.write(text)


def ensure_dir(dir_path: Union[str, Path]):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def safe_load_json(file_path: Union[str, Path], default: Any = None, **kwargs):
    """
    Safely load JSON from a file, returning default if file doesn't exist.
    
    Args:
        file_path: Path to the file
        default: Default value to return if file doesn't exist
        **kwargs: Additional arguments for json.load()
    
    Returns:
        Loaded data or default value
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return default
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f, **kwargs)
    except (json.JSONDecodeError, FileNotFoundError):
        return default


def get_safe_output_path(base_name: str, output_dir: str = "./output", extension: str = ".json") -> Path:
    """
    Get a safe output path with automatic directory creation.
    
    Args:
        base_name: Base name for the file
        output_dir: Output directory (will be created if needed)
        extension: File extension
    
    Returns:
        Path object for the output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_name.endswith(extension):
        base_name += extension
    
    return output_dir / base_name


def get_results_path(task_name: str, model_name: str = None, output_dir: str = "./results") -> Path:
    """
    Get a standardized results file path for evaluation scripts.
    
    Args:
        task_name: Name of the task
        model_name: Optional model name to include in filename
        output_dir: Results directory
    
    Returns:
        Path object for the results file
    """
    if model_name:
        filename = f"results_{task_name}_{model_name}.json"
    else:
        filename = f"results_{task_name}.json"
    
    return get_safe_output_path(filename, output_dir, "")


def get_figures_path(task_name: str, model_name: str = None, output_dir: str = "./figures") -> Path:
    """
    Get a standardized figures file path for evaluation scripts.
    
    Args:
        task_name: Name of the task
        model_name: Optional model name to include in filename
        output_dir: Figures directory
    
    Returns:
        Path object for the figures file
    """
    if model_name:
        filename = f"results_{task_name}_{model_name}.json"
    else:
        filename = f"results_{task_name}.json"
    
    return get_safe_output_path(filename, output_dir, "")


def log_file_operation(operation: str, file_path: Union[str, Path], success: bool = True):
    """
    Log file operations for debugging.
    
    Args:
        operation: Type of operation (save, load, create_dir, etc.)
        file_path: Path involved in the operation
        success: Whether the operation was successful
    """
    level = logging.INFO if success else logging.ERROR
    status = "succeeded" if success else "failed"
    logger.log(level, f"File operation '{operation}' {status}: {file_path}")