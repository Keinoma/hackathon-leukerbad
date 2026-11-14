"""Interfaces shared by robotic arm implementations."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Pose:
    """Cartesian arm pose in millimetres and degrees."""

    x: float
    y: float
    z: float
    rotation: float = 0.0


class RobotArm(Protocol):
    """Protocol for a simple robotic arm controller."""

    def move_to(self, pose: Pose) -> None:
        """Move the arm to a target pose."""

    def open_gripper(self) -> None:
        """Open the gripper."""

    def close_gripper(self) -> None:
        """Close the gripper."""
