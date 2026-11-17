import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

TARGET_FILE = Path(__file__).resolve().parent / "vision_state.json"
ROBOT_DIR = Path(__file__).resolve().parent / "FAIRE BOUGER ROBOT"
SCRIPT = ROBOT_DIR / "pick_rota.py"
MIN_CONFIDENCE = 0.60
MM_TO_M = 1 / 1000.0

POLL_S = 0.2          # période de lecture du json
SETTLE_S = 1.5        # pause après un pick, le temps que la caméra se remette à jour
MAX_FAILURES = 3      # échecs consécutifs avant arrêt complet


class NoTarget(Exception):
    """Pas de cible exploitable pour l'instant (situation normale, on réessaie)."""


def load_target(path=TARGET_FILE):
    """Renvoie (x_mm, y_mm, theta_deg, timestamp) ou lève NoTarget."""
    if not path.exists():
        raise NoTarget(f"{path} introuvable")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        # fichier en cours d'écriture par le script de vision : on réessaiera
        raise NoTarget("JSON illisible (écriture en cours ?)")

    timestamp = None
    if isinstance(data, dict) and "target" in data:
        if data.get("state") != "ready" or not isinstance(data["target"], dict):
            raise NoTarget("aucune puff valide dans l'état de vision")
        timestamp = data.get("timestamp", data.get("created_at_unix"))
        if not isinstance(timestamp, (int, float)) or not 0 <= time.time() - timestamp <= 2:
            raise NoTarget("mesure de vision périmée")
        data = data["target"]

    measures = data if isinstance(data, list) else [data]
    measures = [m for m in measures if m.get("confidence", 0.0) >= MIN_CONFIDENCE]
    if not measures:
        raise NoTarget(f"aucune mesure avec confiance >= {MIN_CONFIDENCE}")

    try:
        x = statistics.median(m["x_mm"] for m in measures)
        y = statistics.median(m["y_mm"] for m in measures)
        theta = statistics.median(m["theta_deg"] for m in measures)
    except KeyError as e:
        raise NoTarget(f"clé manquante : {e}")

    return x, y, theta, timestamp


def run_pick(x_mm, y_mm, theta):
    cmd = [sys.executable, str(SCRIPT),
           f"{x_mm * MM_TO_M:.6f}", f"{y_mm * MM_TO_M:.6f}", f"{theta:.2f}"]
    print("Lancement :", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROBOT_DIR).returncode


def main():
    last_pick_end = 0.0
    failures = 0
    last_msg = None
    print("Surveillance de", TARGET_FILE, "(Ctrl+C pour arrêter)")

    try:
        while True:
            try:
                x_mm, y_mm, theta, ts = load_target()
            except NoTarget as e:
                if str(e) != last_msg:          # évite de spammer la console
                    print("En attente :", e)
                    last_msg = str(e)
                time.sleep(POLL_S)
                continue

            last_msg = None
            if ts is not None and ts <= last_pick_end:
                time.sleep(POLL_S)               # mesure antérieure au dernier pick : ignorée
                continue

            print(f"Cible : x = {x_mm:.2f} mm, y = {y_mm:.2f} mm, theta = {theta:.2f} deg")
            rc = run_pick(x_mm, y_mm, theta)
            last_pick_end = time.time()

            if rc == 0:
                failures = 0
            else:
                failures += 1
                print(f"pick_rota.py a échoué (code {rc}), échec {failures}/{MAX_FAILURES}")
                if failures >= MAX_FAILURES:
                    raise SystemExit("Trop d'échecs consécutifs, arrêt.")

            time.sleep(SETTLE_S)
    except KeyboardInterrupt:
        print("\nArrêt demandé.")


if __name__ == "__main__":
    main()
