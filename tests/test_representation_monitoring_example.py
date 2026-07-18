import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_representation_monitoring_example_fresh_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "representation_monitoring.py"
    environment = {
        **os.environ,
        "VERTABRAE_EXAMPLE_OUTPUT_DIR": str(tmp_path),
    }

    def run(*arguments):
        return subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

    fresh = run("--epochs", "1")
    assert fresh.returncode == 0, fresh.stderr
    history_path = tmp_path / "representation_monitoring.jsonl"
    state_path = tmp_path / "representation_monitoring_state.pt"
    assert history_path.is_file()
    assert state_path.is_file()

    duplicate_fresh = run("--epochs", "1")
    assert duplicate_fresh.returncode != 0
    assert "Use --resume" in duplicate_fresh.stderr

    resumed = run("--resume", "--epochs", "2")
    assert resumed.returncode == 0, resumed.stderr
    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert [record["evaluation_index"] for record in records] == [0, 1]
    assert [record["context"]["epoch"] for record in records] == [0, 1]

    before = history_path.read_bytes()
    complete = run("--resume", "--epochs", "2")
    assert complete.returncode == 0, complete.stderr
    assert history_path.read_bytes() == before

    backup = state_path.with_suffix(".backup")
    state_path.replace(backup)
    missing = run("--resume", "--epochs", "3")
    assert missing.returncode != 0
    assert "missing" in missing.stderr
    backup.replace(state_path)

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    state["global_step"] += 1
    torch.save(state, state_path)
    mismatch = run("--resume", "--epochs", "3")
    assert mismatch.returncode != 0
    assert "coordinates do not match" in mismatch.stderr
    assert history_path.read_bytes() == before
