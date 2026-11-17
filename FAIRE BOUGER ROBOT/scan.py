from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

bus = FeetechMotorsBus(
    port="COM11",
    motors={
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect()
print(bus.sync_read("Present_Position", normalize=False))
bus.disconnect()