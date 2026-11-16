from common import *
from utils import *


robot = connect()
close_gripper(robot)
move_to(robot, WAIT)
close_gripper(robot)
move_to(robot, REST)
close_gripper(robot)

safe_disconnect(robot)
