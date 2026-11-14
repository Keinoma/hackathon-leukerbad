from leukerbad.detection.base import Detection
from leukerbad.robotics.base import Pose


def test_detection_and_pose_data_models() -> None:
    detection = Detection("object", 0.95, (1, 2, 30, 40))
    pose = Pose(100, 200, 150)

    assert detection.label == "object"
    assert detection.bounding_box == (1, 2, 30, 40)
    assert pose.rotation == 0.0
