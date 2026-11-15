"""Couple coupé : place le bras à la main (ex. pose de repos) et lis les angles."""
import time
from common import *
from urdf_ik import SO101Kin

robot = connect()
kin = SO101Kin(URDF)
robot.bus.disable_torque()
print("Bouge le bras à la main. Ctrl+C pour quitter (le bras est libre : soutiens-le).")
try:
    while True:
        q = read_joints(robot)
        x, y, z = kin.fk([q[j] for j in ARM])[:3, 3]
        print({k: round(v, 1) for k, v in q.items()}, f"xyz=({x:.3f},{y:.3f},{z:.3f})   ", end="\r")
        time.sleep(0.2)
except KeyboardInterrupt:
    print()
finally:
    robot.disconnect()
