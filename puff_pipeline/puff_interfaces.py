"""Typed contracts between puff detection, pose estimation and execution.

The public coordinate contract is table-frame millimetres/degrees. Concrete
implementations remain in the existing vision, IK, planner and simulator code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    center_px: tuple[float, float]
    angle_deg: float
    confidence: float
    touching_border: bool = False


@dataclass(frozen=True)
class ObjectEstimate:
    x_mm: float
    y_mm: float
    yaw_deg: float
    uncertainty_mm: float
    confidence: float
    source: str


@dataclass(frozen=True)
class JointSolution:
    measured_deg: tuple[float, ...]
    command_deg: tuple[float, ...]
    tcp_xyz_mm: tuple[float, float, float]
    position_error_mm: float


@dataclass(frozen=True)
class PickPlan:
    target: ObjectEstimate
    waypoint_names: tuple[str, ...]
    joint_solutions: tuple[JointSolution, ...]
    gripper_open: float
    gripper_close: float


@dataclass(frozen=True)
class AttemptResult:
    completed: bool
    grasp_confirmed: bool
    lifted: bool
    physical_evidence: str
    events: tuple[dict, ...] = ()


@dataclass(frozen=True)
class HardwareAuthorization:
    """Run-scoped acknowledgement required by any real-motion adapter."""

    explicitly_enabled: bool


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> Detection | None: ...


class PoseEstimator(Protocol):
    def estimate(self, detection: Detection, camera_pose: Sequence[float]) -> ObjectEstimate: ...


class InverseKinematics(Protocol):
    def solve(self, target: ObjectEstimate, height_mm: float) -> JointSolution: ...


class PickPlanner(Protocol):
    def plan(self, target: ObjectEstimate) -> PickPlan: ...


class MotionExecutor(Protocol):
    def execute(self, plan: PickPlan, authorization: HardwareAuthorization) -> AttemptResult: ...


class Simulator(Protocol):
    def run(self, plan: PickPlan, seed: int) -> AttemptResult: ...
