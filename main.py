import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

TARGET_FILE = Path("target.json")
SCRIPT = "pick_rota.py"
MIN_CONFIDENCE = 0.0     # monte à 0.8 par exemple pour ignorer les mesures douteuses
MM_TO_M = 1 / 1000.0     # pick_rota.py attend des mètres (cf. "x, y = ... # mètres")


def load_target(path=TARGET_FILE):
    """Lit le json et renvoie (x_mm, y_mm, theta_deg)."""
    if not path.exists():
        raise SystemExit(f"{path} introuvable.")

    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path} n'est pas un JSON valide : {e}")

    # Shared vision state: unwrap the current target before reading coordinates.
    if isinstance(data, dict) and "target" in data:
        if data.get("state") != "ready" or not isinstance(data["target"], dict):
            raise SystemExit("Aucune puff valide dans l'état de vision actuel.")
        timestamp = data.get("timestamp", data.get("created_at_unix"))
        if not isinstance(timestamp, (int, float)) or not 0 <= time.time() - timestamp <= 2:
            raise SystemExit("Mesure de vision périmée : relancer ou vérifier la caméra.")
        data = data["target"]

    # Ancien format : dict unique ou liste de mesures.
    measures = data if isinstance(data, list) else [data]
    measures = [m for m in measures if m.get("confidence", 1.0) >= MIN_CONFIDENCE]
    if not measures:
        raise SystemExit(f"Aucune mesure exploitable dans {path} (confiance >= {MIN_CONFIDENCE}).")

    try:
        x = statistics.median(m["x_mm"] for m in measures)
        y = statistics.median(m["y_mm"] for m in measures)
        theta = statistics.median(m["theta_deg"] for m in measures)
    except KeyError as e:
        raise SystemExit(f"Clé manquante dans {path} : {e}")

    print(f"{len(measures)} mesure(s) -> x = {x:.2f} mm, y = {y:.2f} mm, theta = {theta:.2f} deg")
    return x, y, theta


def main():
    x_mm, y_mm, theta = load_target()

    cmd = [
        sys.executable, SCRIPT,
        f"{x_mm * MM_TO_M:.6f}",
        f"{y_mm * MM_TO_M:.6f}",
        f"{theta:.2f}",
    ]
    print("Lancement :", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
