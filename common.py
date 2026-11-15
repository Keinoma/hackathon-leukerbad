import time
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # voir note si ImportError
from urdf_ik import ARM

PORT = "COM11"
ROBOT_ID = "my_follower"             # <- le même id qu'à la calibration
URDF = r"C:\Projets\so101xy\so101_new_calib.urdf"
Z_K = 5.0


REST = {'shoulder_pan': -22.2, 'shoulder_lift': -102.8, 'elbow_flex': 61.5, 'wrist_flex': 92.0, 'wrist_roll': -0.9}
WAIT = {'shoulder_pan': -18.9, 'shoulder_lift': -34.5, 'elbow_flex': 8.9, 'wrist_flex': 69.1, 'wrist_roll': -0.8}
ORIGIN = {'shoulder_pan': -22.4, 'shoulder_lift': 26.0, 'elbow_flex': -1.4, 'wrist_flex': 73.1, 'wrist_roll': -1.0}

def connect():
    # use_degrees=True : angles en degrés comme dans l'URDF
    # max_relative_target=10 : limite chaque commande à 10 deg d'écart avec la position actuelle
    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id=ROBOT_ID, use_degrees=True, max_relative_target=10.0))
    robot.connect(calibrate=False)
    return robot


OFFSETS = {}

# Offset appliqué UNIQUEMENT à la commande
CMD_OFFSETS = {"wrist_flex": 0.0}


def read_joints(robot, raw=False):
    """Angles (degrés). raw=False : angles 'modèle' (lecture + OFFSETS), utilisés par l'IK et les poses."""
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] + (0.0 if raw else OFFSETS.get(j, 0.0)) for j in ARM + ["gripper"]}

def origin_xyz(kin):
    """(x_origin, y_origin, z_table) en repère robot, calculés depuis la pose ORIGIN."""
    if ORIGIN is None:
        raise SystemExit("ORIGIN n'est pas défini dans common.py (utilise record_pose.py, "
                         "pince posée sur le point (0, 0) de la table).")
    return tuple(kin.fk([ORIGIN[j] for j in ARM])[:3, 3])

# Réglages de vitesse
PEAK_SPEED_DEG_S = 90.0   # vitesse de pointe visée (deg/s) : augmente pour aller plus vite
MIN_DURATION_S = 1.0      # durée minimale d'un mouvement (même très court)


def move_to(robot, target, duration_s=None, fps=50):
    """Mouvement fluide (profil 'minimum jerk') des angles actuels vers target (dict nom -> degrés).
    Vitesse et accélération sont nulles au début et à la fin : démarrage et arrêt sans à-coup.
    duration_s=None : la durée est calculée à partir du plus grand déplacement et de PEAK_SPEED_DEG_S."""
    start = read_joints(robot)
    goal = {**start, **target}
    for j in target:                       # offset de commande : seulement sur les joints commandés
        goal[j] += CMD_OFFSETS.get(j, 0.0)
    dmax = max(abs(goal[j] - start[j]) for j in goal)
    if duration_s is None:
        duration_s = max(MIN_DURATION_S, 1.875 * dmax / PEAK_SPEED_DEG_S)  # 1.875 = pic de vitesse / vitesse moyenne
    t0 = time.perf_counter()
    while True:
        t = min(1.0, (time.perf_counter() - t0) / duration_s)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        robot.send_action({f"{j}.pos": start[j] + s * (goal[j] - start[j]) - OFFSETS.get(j, 0.0) for j in goal})
        if t >= 1.0:
            break
        time.sleep(1 / fps)


def safe_disconnect(robot):
    """disconnect() coupe le couple : le bras tombe s'il n'est pas sur sa pose de repos."""
    if REST is not None:
        print("Retour à la pose de repos...")
        move_to(robot, REST)
        time.sleep(0.5)
    else:
        input("REST non défini : TIENS le bras, puis Entrée (le couple va être coupé)...")
    robot.disconnect()
