"""JSON report serialization."""

import json
from pathlib import Path
from typing import Any


def save_json_report(result: Any, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
