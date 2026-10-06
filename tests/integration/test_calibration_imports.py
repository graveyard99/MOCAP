import json

import cv2
import numpy as np
import pytest

from openmocap.calibration import (
    calibrate_intrinsics,
    checkerboard_points,
    detect_charuco,
    import_colmap_text,
    import_opencv,
    import_metashape_xml,
    load_cameras,
    save_cameras,
    static_scene_mask,
)
from openmocap.cameras import look_at
from openmocap.types import Camera, CameraIntrinsics, CameraPose


def test_canonical_camera_json(tmp_path) -> None:
    camera = look_at("A", [0, 2, 5], [0, 1, 0], CameraIntrinsics(1000, 1000, 500, 300, 1000, 600))
    path = tmp_path / "cameras.json"
    save_cameras([camera], path)
    restored = load_cameras(path)
    assert restored[0].to_dict() == camera.to_dict()
    data = json.loads(path.read_text())
    data["units"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="explicit conversion"):
        load_cameras(path)


def test_colmap_text_scale_and_pose(tmp_path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# Cameras\n1 OPENCV 1000 600 1000 1000 500 300 .01 -.002 0 0\n"
    )
    (tmp_path / "images.txt").write_text(
        "# Images\n1 1 0 0 0 0 0 -5 1 CAM_A.png\n\n2 1 0 0 0 -2 0 -5 1 CAM_B.png\n\n"
    )
    cameras = import_colmap_text(tmp_path, metric_scale=0.5)
    assert [c.id for c in cameras] == ["CAM_A", "CAM_B"]
    assert np.allclose(cameras[0].pose.centre, [0, 0, 2.5])
    assert np.allclose(cameras[1].pose.centre, [1, 0, 2.5])
    assert all(c.locked for c in cameras)
    with pytest.raises(ValueError, match="positive"):
        import_colmap_text(tmp_path, metric_scale=0)


def test_opencv_file_storage_import(tmp_path) -> None:
    path = tmp_path / "camera.yml"
    store = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    store.write("K", np.array([[1000.0, 0, 500], [0, 1000, 300], [0, 0, 1]]))
    store.write("D", np.zeros(5))
    store.write("R", np.eye(3))
    store.write("t", np.array([0.0, 0, 5.0]))
    store.write("image_width", 1000)
    store.write("image_height", 600)
    store.release()
    camera = import_opencv(path, "A", world_convention="Y_UP_METRES")
    assert np.allclose(camera.project([0, 0, 0]), [500, 300])
    with pytest.raises(ValueError, match="Declare"):
        import_opencv(path, "A", world_convention="UNKNOWN")


def test_target_intrinsic_calibration_from_known_geometry() -> None:
    intrinsics = CameraIntrinsics(
        900, 920, 640, 360, 1280, 720, [0.03, -0.01, 0.001, -0.001, 0.001]
    )
    points = checkerboard_points(7, 5, 0.04)
    objects, images = [], []
    for i in range(12):
        rotation = cv2.Rodrigues(np.array([0.07 * i - 0.3, 0.04 * i - 0.2, 0.015 * i]))[0]
        pose = CameraPose(rotation, [-0.15 + 0.01 * i, -0.1 + 0.005 * i, 0.7 + 0.03 * i])
        objects.append(points)
        images.append(Camera("target", intrinsics, pose).project(points))
    result = calibrate_intrinsics(objects, images, (1280, 720), target_spacing_metres=0.04)
    assert result.rms_pixels < 0.001
    assert abs(result.intrinsics.fx - intrinsics.fx) < 0.1
    assert abs(result.intrinsics.fy - intrinsics.fy) < 0.1


def test_charuco_target_detection() -> None:
    board = cv2.aruco.CharucoBoard(
        (7, 5), 0.04, 0.025, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    )
    image = board.generateImage((1000, 700), marginSize=25)
    points, pixels = detect_charuco(image, 7, 5, 0.04, 0.025)
    assert len(points) >= 20
    assert points.shape[1] == 3
    assert pixels.shape == (len(points), 2)


def test_metashape_local_metric_transform_and_tangential_order(tmp_path) -> None:
    path = tmp_path / "cameras.xml"
    path.write_text("""<document><chunk><sensors><sensor id="1" type="frame">
      <resolution width="1000" height="600"/><calibration><f>900</f><cx>10</cx><cy>-5</cy>
      <k1>0.1</k1><p1>0.002</p1><p2>-0.003</p2></calibration></sensor></sensors>
      <cameras><camera id="0" label="A" sensor_id="1"><transform>
      1 0 0 2 0 1 0 3 0 0 1 4 0 0 0 1</transform></camera></cameras></chunk></document>""")
    camera = import_metashape_xml(path, metric_scale=0.5)[0]
    assert np.allclose(camera.pose.centre, [1, 1.5, 2])
    assert camera.intrinsics.cx == 510
    assert camera.intrinsics.cy == 295
    assert np.allclose(camera.intrinsics.distortion, [0.1, 0, -0.003, 0.002, 0])


def test_performer_exclusion_static_mask() -> None:
    actor = np.zeros((100, 100), dtype=np.uint8)
    actor[40:60, 40:60] = 255
    static = static_scene_mask(actor, padding_pixels=5)
    assert static[50, 50] == 0
    assert static[36, 50] == 0
    assert static[10, 10] == 255
