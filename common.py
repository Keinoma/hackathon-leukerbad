import time
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # voir note si ImportError
from urdf_ik import ARM

PORT = "COM11"
ROBOT_ID = "my_follower"             # <- le même id qu'à la calibration
URDF = r"C:\Projets\so101xy\so101_new_calib.urdf"

# Pose de repos (bras replié, qui tient tout seul). À remplir avec read_pose.py, par exemple :
# REST = {"shoulder_pan": 0.0, "shoulder_lift": -98.0, "elbow_flex": 96.0,
#         "wrist_flex": 70.0, "wrist_roll": 0.0}
REST = None

# Pose d'attente au-dessus de la scène (optionnelle, à remplir avec record_pose.py)
WAIT = {'shoulder_pan': -14.8, 'shoulder_lift': -113.5, 'elbow_flex': 46.1, 'wrist_flex': -16.8, 'wrist_roll': -5.7}


def connect():
    # use_degrees=True : angles en degrés comme dans l'URDF
    # max_relative_target=10 : limite chaque commande à 10 deg d'écart avec la position actuelle
    robot = SO101Follower(SO101FollowerConfig(
        port=PORT, id=ROBOT_ID, use_degrees=True, max_relative_target=10.0))
    robot.connect(calibrate=False)
    return robot


# Corrections d'offset (degrés) : angle_modèle = lecture_robot + OFFSET.
OFFSETS = {} # à mesurer avec calib_vertical.py

# Offset appliqué UNIQUEMENT à la commande 
CMD_OFFSETS = {}


def read_joints(robot, raw=False):
    """Angles (degrés). raw=False : angles 'modèle' (lecture + OFFSETS), utilisés par l'IK et les poses."""
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] + (0.0 if raw else OFFSETS.get(j, 0.0)) for j in ARM + ["gripper"]}


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
