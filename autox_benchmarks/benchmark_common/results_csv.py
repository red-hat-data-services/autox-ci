"""Write benchmark result rows to CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_results_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
