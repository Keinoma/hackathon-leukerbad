"""Test the integration reader without importing or running robot hardware code."""

import importlib.util
import json
from pathlib import Path
import time

import pytest


def reader():
    spec = importlib.util.spec_from_file_location("robot_entry", Path(__file__).parents[1] / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reader_accepts_live_shared_state(tmp_path):
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"state": "ready", "timestamp": time.time(),
                               "target": {"x_mm": 20, "y_mm": -30,
                                          "theta_deg": 4, "confidence": .9}}))
    assert reader().load_target(path) == (20, -30, 4)


@pytest.mark.parametrize("state,timestamp,target", [
    ("empty", 0, None),
    ("ready", 0, {"x_mm": 20, "y_mm": -30, "theta_deg": 4}),
])
def test_reader_rejects_missing_or_stale_targets(tmp_path, state, timestamp, target):
    path = tmp_path / "target.json"
    path.write_text(json.dumps({"state": state, "timestamp": timestamp, "target": target}))
    with pytest.raises(SystemExit):
        reader().load_target(path)
