"""Validated, read-only loader for the project's shared configuration."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "project_config.json"


def load_project_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != "puff_sort_project_v1":
        raise ValueError("Schéma de configuration puff-sort inconnu.")
    if config.get("units") != {"length": "mm", "angle": "deg"}:
        raise ValueError("Les unités canoniques doivent être mm et degrés.")
    if config.get("frames", {}).get("table", {}).get("axes") != [
        "x_right", "y_away_from_base", "z_up"
    ]:
        raise ValueError("Convention d'axes table incohérente.")
    if config.get("hardware_execution", {}).get("enabled_by_default") is not False:
        raise ValueError("L'accès matériel doit rester désactivé par défaut.")
    for group in ("canonical_components", "calibrations"):
        for relative in config.get(group, {}).values():
            if not (ROOT / relative).is_file():
                raise ValueError(f"Fichier configuré introuvable : {relative}")
    return config
