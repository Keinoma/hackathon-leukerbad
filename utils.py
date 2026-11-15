"""Petites fonctions qui modifient directement l'angle d'un moteur.

Toutes passent par move_to() de common.py : mouvement fluide, offsets de commande
et limite max_relative_target déjà gérés. Angles en degrés ("modèle", comme read_joints).
La pince est sur l'échelle 0-100 de LeRobot (0 = fermée, 100 = ouverte).
"""

import time
import numpy as np
from common import *
import common as _common   # pour lire PEAK_SPEED_DEG_S au moment de l'appel

JOINTS = list(ARM) + ["gripper"]

# Valeurs par défaut de la pince -> À AJUSTER selon ton robot / l'objet
GRIPPER_OPEN = 50.0
GRIPPER_CLOSED = 0.0
GRIPPER_DUREE = 1.0     # durée (s) d'une ouverture / fermeture


# ---------- génériques ----------

def get_joint(robot, joint):
    """Angle actuel (degrés) d'un moteur."""
    _check(joint)
    return read_joints(robot)[joint]


def set_joint(robot, joint, angle, duration_s=None):
    """Amène UN moteur à l'angle `angle` (degrés), les autres ne bougent pas."""
    _check(joint)
    move_to(robot, {joint: float(angle)}, duration_s=duration_s)


def move_joint_by(robot, joint, delta, duration_s=None):
    """Tourne UN moteur de `delta` degrés par rapport à sa position actuelle."""
    _check(joint)
    set_joint(robot, joint, get_joint(robot, joint) + delta, duration_s)


def set_joints(robot, angles, duration_s=None):
    """Amène plusieurs moteurs en même temps : angles = {nom: degrés}."""
    for j in angles:
        _check(j)
    move_to(robot, {j: float(a) for j, a in angles.items()}, duration_s=duration_s)


# ---------- un raccourci par moteur ----------

def set_shoulder_pan(robot, angle, duration_s=None):
    set_joint(robot, "shoulder_pan", angle, duration_s)


def set_shoulder_lift(robot, angle, duration_s=None):
    set_joint(robot, "shoulder_lift", angle, duration_s)


def set_elbow_flex(robot, angle, duration_s=None):
    set_joint(robot, "elbow_flex", angle, duration_s)


def set_wrist_flex(robot, angle, duration_s=None):
    set_joint(robot, "wrist_flex", angle, duration_s)


def set_wrist_roll(robot, angle, duration_s=None):
    set_joint(robot, "wrist_roll", angle, duration_s)


# ---------- pince ----------

def set_gripper(robot, value, duration_s=GRIPPER_DUREE):
    set_joint(robot, "gripper", value, duration_s)


def open_gripper(robot, value=None, duration_s=GRIPPER_DUREE):
    set_gripper(robot, GRIPPER_OPEN if value is None else value, duration_s)


def close_gripper(robot, value=None, duration_s=GRIPPER_DUREE):
    set_gripper(robot, GRIPPER_CLOSED if value is None else value, duration_s)


# ---------- interne ----------

def _check(joint):
    if joint not in JOINTS:
        raise ValueError(f"moteur inconnu : {joint!r} (attendu : {JOINTS})")


# ----------- le long axe vertical ----------

def move_z(robot, kin, dz, steps=20, duration_s=None, z_min=None, max_err=0.005):
    """Déplace la pince en ligne droite le long de z de `dz` mètres (dz < 0 : descend, dz > 0 : monte).
    x, y, wrist_roll et la correction de wrist_flex (WRIST_K) restent ceux de la pose actuelle.
    z_min : z (repère robot) à ne jamais dépasser vers le bas, par ex. z_table - 0.005."""
    q = read_joints(robot)
    q_now = np.array([q[j] for j in ARM])
    p0 = kin.fk(q_now)[:3, 3]
    if z_min is not None and p0[2] + dz < z_min:
        raise ValueError(f"REFUSÉ : z final {p0[2] + dz:.3f} m < z_min {z_min:.3f} m")

    # On calcule les waypoints par IK, puis on les applique comme DIFFÉRENCE par rapport à la pose actuelle :
    # ainsi roll, WRIST_K et les petits écarts de la pose actuelle sont conservés.
    base, _, _ = kin.ik_down(list(p0), q_now)
    base = np.array(base, dtype=float)
    seed, traj = base, [q_now]
    for k in range(1, steps + 1):
        p = p0 + np.array([0.0, 0.0, dz * k / steps])
        sol, err, tilt = kin.ik_down(list(p), seed)
        if err > max_err:
            raise ValueError(f"REFUSÉ : erreur IK {err*1000:.1f} mm à z = {p[2]:.3f} m")
        if tilt > MAX_TILT_DEG:
            raise ValueError(f"REFUSÉ : pince inclinée de {tilt:.1f} deg à z = {p[2]:.3f} m")
        seed = np.array(sol, dtype=float)
        traj.append(q_now + (seed - base))
    traj = np.array(traj)

    if duration_s is None:
        dmax = np.max(np.abs(traj[-1] - traj[0]))
        duration_s = max(_common.MIN_DURATION_S, 1.875 * dmax / _common.PEAK_SPEED_DEG_S)

    t0 = time.perf_counter()
    while True:
        t = min(1.0, (time.perf_counter() - t0) / duration_s)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5          # même profil que move_to
        f = s * steps
        i = min(int(f), steps - 1)
        qk = traj[i] + (f - i) * (traj[i + 1] - traj[i])
        action = {f"{j}.pos": qk[n] - OFFSETS.get(j, 0.0) for n, j in enumerate(ARM)}
        action["gripper.pos"] = q["gripper"] - OFFSETS.get("gripper", 0.0)
        robot.send_action(action)
        if t >= 1.0:
            break
        time.sleep(1 / 50)


def descend_z(robot, kin, hauteur, **kw):
    """Descend de `hauteur` mètres (valeur positive)."""
    move_z(robot, kin, -abs(hauteur)+0.007, **kw)


def rise_z(robot, kin, hauteur, **kw):
    """Monte de `hauteur` mètres (valeur positive)."""
    move_z(robot, kin, abs(hauteur), **kw)



if __name__ == "__main__":
    print(np.arcsin(1))
    print(np.arcsin(1.5))
