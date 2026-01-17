"""Path resolution utilities with environment variable support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def resolve_path(path: str | Path, expand_user: bool = True) -> Path:
    """
    Resolve a path, expanding environment variables and user home directory.

    Args:
        path: Path string that may contain environment variables (e.g., ${VAR})
        expand_user: Whether to expand ~ to user home directory

    Returns:
        Resolved Path object
    """
    path_str = str(path)

    # Expand environment variables in ${VAR} format
    path_str = os.path.expandvars(path_str)

    # Create Path object
    resolved = Path(path_str)

    # Expand user home directory
    if expand_user:
        resolved = resolved.expanduser()

    return resolved


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_env_or_default(env_var: str, default: str) -> str:
    """Get environment variable or return default value."""
    return os.environ.get(env_var, default)
