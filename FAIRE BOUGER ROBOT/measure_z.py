import time
import numpy as np
from common import *
from urdf_ik import SO101Kin

robot = connect()
kin = SO101Kin(URDF)
robot.bus.disable_torque()   # bras libre : bouge-le à la main
print("Pose le bout de la pince sur la table. Ctrl+C pour quitter.")
try:
    while True:
        q = read_joints(robot)
        x, y, z = kin.fk([q[j] for j in ARM])[:3, 3]
        print(f"x={x:.3f}  y={y:.3f}  z={z:.3f}  inclinaison pince={kin.tilt_deg([q[j] for j in ARM]):.0f} deg   ", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    robot.disconnect()
