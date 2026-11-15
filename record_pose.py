"""Enregistre une pose : couple coupé, place le bras à la main, appuie sur Entrée,
puis copie la ligne affichée dans common.py (REST = ... ou WAIT = ...)."""
import time
from common import *
from urdf_ik import SO101Kin

robot = connect()
kin = SO101Kin(URDF)
robot.bus.disable_torque()
print("Couple coupé : SOUTIENS le bras. Place-le dans la pose voulue.")
print("Entrée = enregistrer la pose, q + Entrée = quitter.")
try:
    while True:
        if input("> ").strip().lower() == "q":
            break
        q = read_joints(robot)
        x, y, z = kin.fk([q[j] for j in ARM])[:3, 3]
        d = {j: round(float(q[j]), 1) for j in ARM}
        print(f"xyz = ({x:.3f}, {y:.3f}, {z:.3f}), gripper = {q['gripper']:.1f}")
        print("A COPIER :  POSE =", d)
finally:
    robot.disconnect()
