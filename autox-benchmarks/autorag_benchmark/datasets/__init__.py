"""
Dataset providers: each prepares a knowledge base dir and benchmark JSON.

To add a dataset: implement prepare(kb_dir, bench_path, **options) -> (num_docs, num_entries),
then register in REGISTRY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# Type: (kb_dir, bench_path, **options) -> (num_docs, num_entries)
PrepareFn = Callable[..., tuple[int, int]]


REGISTRY: dict[str, tuple[PrepareFn, dict]] = {}
# name -> (prepare_function, default_options)


def register(name: str, prepare_fn: PrepareFn, default_options: dict | None = None) -> None:
    """Register a dataset provider."""
    REGISTRY[name] = (prepare_fn, default_options or {})


def get(name: str) -> tuple[PrepareFn, dict]:
    """Get (prepare_fn, default_options) for a dataset. Raises KeyError if unknown."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(REGISTRY.keys())


# Register built-in datasets (import triggers registration)
from autorag_benchmark.datasets import beir  # noqa: E402, F401
from autorag_benchmark.datasets import open_ragbench  # noqa: E402, F401
from autorag_benchmark.datasets import slidevqa  # noqa: E402, F401

__all__ = [
    "REGISTRY",
    "register",
    "get",
    "list_datasets",
    "PrepareFn",
]
