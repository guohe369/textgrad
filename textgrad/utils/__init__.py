"""
Utilities for TextGrad.
"""

from .file_utils import (
    safe_open,
    safe_save_json,
    safe_save_text,
    safe_load_json,
    ensure_dir,
    get_safe_output_path,
    get_results_path,
    get_figures_path,
    log_file_operation,
)

__all__ = [
    "safe_open",
    "safe_save_json", 
    "safe_save_text",
    "safe_load_json",
    "ensure_dir",
    "get_safe_output_path",
    "get_results_path",
    "get_figures_path",
    "log_file_operation",
]