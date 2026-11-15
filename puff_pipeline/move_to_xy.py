"""Move the calibrated SO-101 follower to a table-frame XY test point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from so101_ik import JOINT_NAMES, IKResult, SO101CalibratedIK, verify_calibration_hash

ROOT = Path(__file__).resolve().parent


def move_to_xy(
    robot,
    x_mm: float,
    y_mm: float,
    z_mm: float = 50.0,
    speed_deg_per_s: float = 20.0,
) -> IKResult:
    """Move a connected, calibrated follower to table coordinates (x, y, z).

    The caller must obtain operator authorization and enable follower torque first.
    Coordinates use the kash table frame: +X right, +Y away from the base, +Z up.
    """
    if not 10.0 <= speed_deg_per_s <= 40.0:
        raise ValueError("speed_deg_per_s must be between 10 and 40 degrees/s")

    verify_calibration_hash()
    solution = SO101CalibratedIK().solve(x_mm, y_mm, z_mm)
    bus = robot.bus
    actual = bus.sync_read("Present_Position", num_retry=2)
    if not all(np.isfinite(float(actual[name])) for name in JOINT_NAMES):
        raise RuntimeError("Invalid motor position reading")

    start = np.array([float(actual[name]) for name in JOINT_NAMES])
    target = start.copy()
    target[:5] = solution.command_deg
    duration = max(0.5, float(np.max(np.abs(target[:5] - start[:5]))) / speed_deg_per_s)
    began = time.monotonic()
    while True:
        elapsed = time.monotonic() - began
        alpha = min(1.0, elapsed / duration)
        blend = alpha * alpha * (3.0 - 2.0 * alpha)
        command = start + blend * (target - start)
        robot.send_action({f"{name}.pos": float(value) for name, value in zip(JOINT_NAMES, command)})
        if alpha >= 1.0:
            break
        time.sleep(1.0 / 30.0)
    return solution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x_mm", type=float, help="X table in mm (+X is right).")
    parser.add_argument("y_mm", type=float, help="Y table in mm (+Y is away from the base).")
    parser.add_argument("--z-mm", type=float, default=50.0, help="Height above table origin (default: 50 mm).")
    parser.add_argument("--port", help="Follower serial port, e.g. COM5 or /dev/ttyACM0.")
    parser.add_argument("--robot-id", default="mon_bras", help="LeRobot follower calibration ID.")
    parser.add_argument("--speed", type=float, default=20.0, help="Motion speed in degrees/s (10 to 40).")
    parser.add_argument("--dry-run", action="store_true", help="Solve and validate the target without hardware access.")
    parser.add_argument("--hardware", action="store_true", help="Explicitly allow access to the physical robot.")
    args = parser.parse_args()

    if not 10.0 <= args.speed <= 40.0:
        parser.error("--speed must be between 10 and 40 degrees/s.")
    verify_calibration_hash()
    solution = SO101CalibratedIK().solve(args.x_mm, args.y_mm, args.z_mm)
    print(
        f"Target XY=({args.x_mm:+.1f}, {args.y_mm:+.1f}) mm, Z={args.z_mm:.1f} mm; "
        f"IK residual={solution.pos_error_mm:.2f} mm; "
        f"joints={np.round(solution.state_deg, 1).tolist()}"
    )
    if args.dry_run:
        print("Dry-run complete; no port or motors opened.")
        return
    if not args.hardware:
        parser.error("Hardware access is disabled. Add --hardware to enable the physical follower.")
    if not args.port:
        parser.error("--port is required to command the follower.")

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise SystemExit("LeRobot is not installed in this Python environment.") from exc

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            calibration_dir=ROOT / "calibrations/follower",
            max_relative_target=6.0,
        )
    )
    bus = robot.bus
    activated = False
    try:
        bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Follower calibration is missing or does not match mon_bras.json.")
        for name in JOINT_NAMES:
            if bus.read("Operating_Mode", name, normalize=False, num_retry=2) != 0:
                raise RuntimeError(f"Motor {name} must be in position mode.")

        initial = bus.sync_read("Present_Position", num_retry=2)
        if not all(np.isfinite(float(initial[name])) for name in JOINT_NAMES):
            raise RuntimeError("Invalid motor position reading.")
        bus.sync_write("Goal_Position", {name: float(initial[name]) for name in JOINT_NAMES}, num_retry=2)
        print("Clear the work area and make sure the follower is in its validated safe start pose.")
        if input("Type MOVE to enable torque and move to the requested point: ").strip() != "MOVE":
            print("Cancelled; no movement command sent.")
            return

        recheck = bus.sync_read("Present_Position", num_retry=2)
        if max(abs(float(recheck[name]) - float(initial[name])) for name in JOINT_NAMES[:5]) > 6.0:
            raise RuntimeError("Arm moved during confirmation; restart from the safe start pose.")
        bus.enable_torque(num_retry=3)
        activated = True
        result = move_to_xy(robot, args.x_mm, args.y_mm, args.z_mm, args.speed)
        print(
            f"Arrived at requested table point ({args.x_mm:+.1f}, {args.y_mm:+.1f}, "
            f"{args.z_mm:.1f}) mm. IK residual: {result.pos_error_mm:.2f} mm."
        )
        print("Visually compare the gripper TCP with a reference directly above that marked point.")
        verdict = input("Is the mapping correct at this point? [y/N]: ").strip().lower()
        if verdict in ("y", "yes"):
            print("Point validated. Repeat at other points to check the whole working area.")
        else:
            print("Point not validated. Do not rely on this mapping until it is recalibrated.")
    finally:
        if bus.is_connected:
            if activated:
                try:
                    measured = bus.sync_read("Present_Position", num_retry=2)
                    measured["gripper"] = bus.read("Goal_Position", "gripper", num_retry=2)
                    bus.sync_write("Goal_Position", measured, num_retry=2)
                finally:
                    bus.disconnect(disable_torque=False)
            else:
                bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()