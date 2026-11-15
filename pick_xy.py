"""Saisit un objet en (x, y) puis revient à WAIT.

Usage : python pick_xy.py 0.02 -0.01
Coordonnées en mètres, dans le repère de table de go_xy.py.
Vérifier GRIPPER_OPEN et GRIPPER_CLOSED avant utilisation.
"""
import sys
import time
import numpy as np
import common
from common import *
from urdf_ik import SO101Kin

# Identiques à go_xy.py
Z_TABLE = 0.01
X_ORIGIN = 0.232
Y_ORIGIN = 0.03
MARGE = 0.02
MAX_DELTA_DEG = 270
MAX_TILT_DEG = 3.0
DUREE = None

# Valeurs indicatives en degrés LeRobot : à confirmer sur VOTRE pince.
# La fermeture complète peut être bloquée par l'objet : ne pas forcer.
GRIPPER_OPEN = 40.0
GRIPPER_CLOSED = 0.0
GRIPPER_DURATION = 1.5


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage : python pick_xy.py <x_m> <y_m>")
    x, y = map(float, sys.argv[1:3])
    kin = SO101Kin(URDF)
    robot = connect()
    try:
        if common.WAIT is not None:
            input("Entrée pour aller à WAIT (Ctrl+C pour annuler)...")
            move_to(robot, common.WAIT)
            time.sleep(0.5)

        q = read_joints(robot)
        q_now = np.array([q[j] for j in ARM])
        p_now = kin.fk(q_now)[:3, 3]
        target = [X_ORIGIN + x, Y_ORIGIN + y, Z_TABLE + MARGE]
        print(f"Point demandé (table) = ({x:.3f}, {y:.3f}) -> repère robot = ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")

        sol, err, tilt = kin.ik_down(target, q_now)
        delta = sol - q_now
        print("Pose actuelle (deg):", dict(zip(ARM, np.round(q_now, 1))), "xyz =", np.round(p_now, 3))
        print("Pose cible    (deg):", dict(zip(ARM, np.round(sol, 1))), "xyz =", np.round(target, 3))
        print("Variation     (deg):", dict(zip(ARM, np.round(delta, 1))),
              f"| erreur IK {err*1000:.2f} mm | inclinaison pince {tilt:.1f} deg")
        print(f"Variation max : {np.max(np.abs(delta)):.1f} deg (limite {MAX_DELTA_DEG})")
        if err > 0.005:
            raise SystemExit(f"REFUSÉ : cible hors de portée (erreur IK {err*1000:.1f} mm).")
        if tilt > MAX_TILT_DEG:
            raise SystemExit(f"REFUSÉ : pince inclinée de {tilt:.1f} deg (> {MAX_TILT_DEG}).")
        if np.max(np.abs(delta)) > MAX_DELTA_DEG:
            raise SystemExit(f"REFUSÉ : variation d'angle > {MAX_DELTA_DEG} deg. Rapproche WAIT.")

        input("Entrée pour ouvrir la pince et saisir l'objet (Ctrl+C pour annuler)...")
        print("Ouverture de la pince...")
        move_to(robot, {"gripper": GRIPPER_OPEN}, duration_s=GRIPPER_DURATION)
        time.sleep(0.3)

        print("Déplacement vers l'objet...")
        move_to(robot, dict(zip(ARM, sol)), duration_s=DUREE)
        time.sleep(0.5)

        print("Fermeture de la pince...")
        move_to(robot, {"gripper": GRIPPER_CLOSED}, duration_s=GRIPPER_DURATION)
        time.sleep(0.5)

        if common.WAIT is not None:
            print("Retour à WAIT, pince fermée...")
            move_to(robot, common.WAIT)
        else:
            print("WAIT non défini : le bras reste à la position de saisie.")
        input("Entrée pour terminer (retour à REST si défini, puis coupure du couple)...")
    finally:
        safe_disconnect(robot)


if __name__ == "__main__":
    main()
