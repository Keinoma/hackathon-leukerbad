"""python go_pose.py REST   ou   python go_pose.py WAIT"""
import sys, time
import numpy as np
import common
from common import *

name = sys.argv[1].upper()
pose = getattr(common, name, None)
if pose is None:
    raise SystemExit(f"{name} n'est pas défini dans common.py (utilise record_pose.py).")

robot = connect()
try:
    q = read_joints(robot)
    delta = {j: round(pose[j] - q[j], 1) for j in ARM}
    print("pose actuelle:", {j: round(q[j], 1) for j in ARM})
    print("pose", name, ":", pose)
    print("variation (deg):", delta)
    input("Entrée pour bouger (Ctrl+C pour annuler)...")
    move_to(robot, pose)
    time.sleep(1.0)
    if name != "REST":
        input("Entrée pour terminer (le bras reviendra au repos si REST est défini)...")
finally:
    safe_disconnect(robot)
