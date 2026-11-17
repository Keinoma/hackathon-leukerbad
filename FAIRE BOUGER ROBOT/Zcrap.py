import numpy as np
import sys
from urdf_ik import SO101Kin
from common import *

ANGLE_given = float(sys.argv[3])
x, y = float(sys.argv[1]), float(sys.argv[2])   # mètres
kin = SO101Kin(URDF)
x_org, y_org, z_table = origin_xyz(kin)
X_ORIGIN = 0.217
Y_ORIGIN = 0.07

# Conversion des coordonnées dans le repère robot
x_robot = X_ORIGIN + x
y_robot = Y_ORIGIN + y

ANGLE_rayon = np.degrees(np.arctan2(y_robot, x_robot))
ANGLE_to_assign = ANGLE_given - ANGLE_rayon

print(f"Angle demandé : {ANGLE_given:.2f}°")
print(f"Angle du rayon : {ANGLE_rayon:.2f}°")
print(f"Angle relatif de la pince : {ANGLE_to_assign:.2f}°")
print(f"{x_org}, {y_org}")