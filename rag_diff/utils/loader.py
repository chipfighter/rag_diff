"""Dynamic adapter loading utilities.

Supports loading callables from both Python module paths (e.g. 'my_pkg.adapter:ask')
and file system paths (e.g. 'path/to/adapter.py:ask').
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable


def parse_target(target: str) -> tuple[str, str]:
    """Parse a target string into (module_ref, function_name).

    Accepted formats:
        - 'module:function'
        - 'package.module:function'
        - 'path/to/file.py:function'
    """
    if ":" not in target:
        raise ValueError(
            f"Target '{target}' must be in 'module:function' format "
            f"(e.g., 'my_adapter:ask_rag')"
        )

    module_part, func_part = target.rsplit(":", 1)

    if not module_part.strip():
        raise ValueError(f"Module part cannot be empty in target '{target}'")
    if not func_part.strip():
        raise ValueError(f"Function part cannot be empty in target '{target}'")

    return module_part, func_part


def _is_file_path(module_ref: str) -> bool:
    """Determine if a module reference looks like a file path."""
    return module_ref.endswith(".py") or "/" in module_ref or "\\" in module_ref


def _load_module_from_path(file_path: str):
    """Load a Python module from a file system path."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise ImportError(f"Adapter file not found: {path}")

    module_name = f"_rag_diff_adapter_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_adapter(target: str) -> Callable:
    """Load an adapter callable from a target string.

    Supports both Python module paths and file system paths.
    Validates that the resolved attribute is callable.
    """
    module_ref, func_name = parse_target(target)

    if _is_file_path(module_ref):
        module = _load_module_from_path(module_ref)
    else:
        try:
            module = importlib.import_module(module_ref)
        except ModuleNotFoundError:
            raise ImportError(f"Module '{module_ref}' not found")

    if not hasattr(module, func_name):
        raise AttributeError(
            f"Module '{module_ref}' has no attribute '{func_name}'"
        )

    obj = getattr(module, func_name)
    if not callable(obj):
        raise TypeError(
            f"'{module_ref}:{func_name}' is not callable "
            f"(got {type(obj).__name__})"
        )

    return obj
