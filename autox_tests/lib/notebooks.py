"""Shared notebook execution helpers for AutoML and AutoRAG test suites."""

import functools
import subprocess
import sys
from pathlib import Path

NOTEBOOK_KERNEL_NAME = "autox-notebook-runner"


@functools.lru_cache(maxsize=None)
def ensure_notebook_kernel_registered() -> None:
    """Register the current Python interpreter as a Jupyter kernel (once per process).

    Ensures papermill executes notebooks in the same environment as the tests,
    so packages installed via test extras (e.g. ai4rag, autogluon) are already
    importable and notebook install guards (try/except ImportError) skip pip install.
    """
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ipykernel",
                "install",
                "--user",
                f"--name={NOTEBOOK_KERNEL_NAME}",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"Failed to register Jupyter kernel '{NOTEBOOK_KERNEL_NAME}': {stderr}"
        ) from e


def cleanup_notebook_kernel() -> None:
    """Remove the registered test kernel spec from the user's Jupyter data directory.

    Safe to call even if the kernel was never registered.
    """
    import shutil

    try:
        from jupyter_core.paths import jupyter_data_dir

        kernel_dir = Path(jupyter_data_dir()) / "kernels" / NOTEBOOK_KERNEL_NAME
        shutil.rmtree(kernel_dir, ignore_errors=True)
    except Exception:
        pass
