import sys, time
import numpy as np
from common import *
from urdf_ik import SO101Kin


Z_TABLE = 0.0        # <- valeur mesurée avec measure_z.py
X_ORIGIN = 0.232     # <- position (repère du robot, en m) qui devient le point (0, 0) de ta table
Y_ORIGIN = 0.03       # <- à mesurer avec measure_z.py (pince posée sur ton point d'origine)
MARGE = 0.0        # 2 cm au-dessus de la table
MAX_DELTA_DEG = 270  # refuse si un angle doit varier de plus que ça entre la pose de départ (WAIT) et la cible
MAX_TILT_DEG = 3.0   # écart toléré par rapport à la verticale (pince pointée vers le bas)
DUREE = None         # None = durée automatique (voir PEAK_SPEED_DEG_S dans common.py)

x, y = float(sys.argv[1]), float(sys.argv[2])   # mètres

kin = SO101Kin(URDF)
robot = connect()
import common
try:
    if common.WAIT is not None:
        input("Entrée pour aller à la pose d'attente WAIT (Ctrl+C pour annuler)...")
        move_to(robot, common.WAIT)
        time.sleep(0.5)

    q = read_joints(robot)
    q_now = np.array([q[j] for j in ARM])
    p_now = kin.fk(q_now)[:3, 3]
    target = [X_ORIGIN + x, Y_ORIGIN + y, Z_TABLE + MARGE]   # repère robot
    print(f"point demandé (repère table) = ({x:.3f}, {y:.3f})  ->  repère robot = ({target[0]:.3f}, {target[1]:.3f})")

    sol, err, tilt = kin.ik_down(target, q_now)
    delta = sol - q_now
    print("pose actuelle (deg):", dict(zip(ARM, np.round(q_now, 1))), "xyz =", np.round(p_now, 3))
    print("pose cible    (deg):", dict(zip(ARM, np.round(sol, 1))), "xyz =", np.round(target, 3))
    print("variation     (deg):", dict(zip(ARM, np.round(delta, 1))), f"| erreur IK {err*1000:.2f} mm | inclinaison pince {tilt:.1f} deg")

    print(f"variation max : {np.max(np.abs(delta)):.1f} deg (limite {MAX_DELTA_DEG})")
    if err > 0.005:
        raise SystemExit(f"REFUSÉ : cible hors de portée (erreur IK {err*1000:.1f} mm). Essaie un autre (x, y) ou un z plus haut.")
    if tilt > MAX_TILT_DEG:
        raise SystemExit(f"REFUSÉ : pince inclinée de {tilt:.1f} deg (> {MAX_TILT_DEG}) : cible trop loin ou trop proche "
                         "pour garder la pince verticale. Distance à la base conseillée : environ 0,12 à 0,29 m.")
    if np.max(np.abs(delta)) > MAX_DELTA_DEG:
        raise SystemExit(f"REFUSÉ : un angle doit varier de plus de {MAX_DELTA_DEG} deg entre la pose actuelle "
                         "et la cible. Rapproche WAIT de la zone de travail (record_pose.py) "
                         "ou augmente MAX_DELTA_DEG.")

    input("Entrée pour bouger (Ctrl+C pour annuler)...")
    move_to(robot, dict(zip(ARM, sol)), duration_s=DUREE)
    time.sleep(1.0)
    if common.WAIT is not None:
        move_to(robot, common.WAIT)
    input("Entrée pour terminer (le bras reviendra au repos)...")
finally:
    safe_disconnect(robot)
