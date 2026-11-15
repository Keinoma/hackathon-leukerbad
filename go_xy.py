import sys, time
import numpy as np
from common import *
from urdf_ik import SO101Kin



x, y = float(sys.argv[1]), float(sys.argv[2])   # mètres
kin = SO101Kin(URDF)
x_org, y_org, z_table = origin_xyz(kin)   # repère robot, déduit de la pose ORIGIN
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
    target = [x_org + x, y_org + y, z_table + MARGE - Z_K * (x**2 + y**2)]   # repère robot
    print(f"point demandé (repère table) = ({x:.3f}, {y:.3f})  ->  repère robot = ({target[0]:.3f}, {target[1]:.3f})")

    # --- vérifications de cohérence (après une recalibration, les anciennes valeurs ne valent plus rien) ---
    for j, a, lo, hi in zip(ARM, q_now, kin.lo, kin.hi):
        if a < lo - 5 or a > hi + 5:
            print(f"ATTENTION : {j} lit {a:.1f} deg, hors des limites du modèle [{lo:.0f}, {hi:.0f}] : "
                  "calibration incohérente ou OFFSETS périmés.")
    if np.hypot(target[0], target[1]) > 0.5 or abs(target[2]) > 0.5:
        raise SystemExit(f"REFUSÉ : la cible est à {np.hypot(target[0], target[1])*1000:.0f} mm de la base. "
                         "Le bras ne dépasse pas ~400 mm : vérifie que ORIGIN et tes arguments "
                         "sont bien en MÈTRES.")

    sol, err, tilt = kin.ik_down(target, q_now)
    sol = np.array(sol, dtype=float)
    wrist_corr = WRIST_K * (x**2 + y**2)
    sol[ARM.index("wrist_flex")] += wrist_corr
    print(f"correction wrist_flex = {wrist_corr:+.2f} deg (WRIST_K = {WRIST_K})")
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