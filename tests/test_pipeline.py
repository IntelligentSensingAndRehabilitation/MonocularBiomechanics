import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pytest

N_JOINTS = 87  # bml_movi_87 skeleton
TEST_VIDEO = Path(__file__).parent.parent / "data" / "running_sagittal.mp4"
OVERLAY_MAX_FRAMES = 10  # limit overlay rendering for test speed


# ---------- session fixtures: run the full pipeline once ----------


@pytest.fixture(scope="session")
def test_video():
    assert TEST_VIDEO.exists(), f"Test video not found: {TEST_VIDEO}"
    return str(TEST_VIDEO)


@pytest.fixture(scope="session")
def detected_keypoints(test_video):
    """Run actual Metrabs detection on the test video."""
    from monocular_demos.utils import load_metrabs, video_reader

    model = load_metrabs()
    vid, n_frames = video_reader(test_video, batch_size=2)

    pose3d_list, pose2d_list, confs_list = [], [], []
    for frame_batch in vid:
        pred = model.detect_poses_batched(frame_batch, skeleton="bml_movi_87")
        for i in range(len(frame_batch)):
            box = pred["boxes"][i]
            if len(box) == 0:
                pose3d_list.append(np.zeros((N_JOINTS, 3)))
                pose2d_list.append(np.zeros((N_JOINTS, 2)))
                confs_list.append(np.zeros(N_JOINTS))
            else:
                pose3d_list.append(pred["poses3d"][i][0].numpy())
                pose2d_list.append(pred["poses2d"][i][0].numpy())
                confs_list.append(np.ones(N_JOINTS))

    return (
        np.stack(pose3d_list),   # (n_frames, 87, 3)
        np.stack(pose2d_list),   # (n_frames, 87, 2)
        np.stack(confs_list),    # (n_frames, 87)
    )


@pytest.fixture(scope="session")
def fitted_model(test_video, detected_keypoints):
    """Build dataset from detected keypoints and run biomechanical fitting."""
    import cv2
    from monocular_demos.dataset import MonocularDataset, get_samsung_calibration
    from monocular_demos.biomechanics_mjx.monocular_trajectory import get_model, fit_model
    from monocular_demos.utils import joint_names

    kp3d, kp2d, confs = detected_keypoints
    cap = cv2.VideoCapture(test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    n_frames = len(kp3d)
    timestamps = np.arange(n_frames) / fps

    dataset = MonocularDataset(
        timestamps=[timestamps],
        keypoints_2d=[kp2d[np.newaxis]],        # (1, n_frames, 87, 2)
        keypoints_3d=[kp3d[np.newaxis]],        # (1, n_frames, 87, 3)
        keypoint_confidence=[confs[np.newaxis]], # (1, n_frames, 87)
        camera_params=get_samsung_calibration(),
        phone_attitude=None,
    )

    model = get_model(dataset, joint_names=joint_names)
    fitted, metrics = fit_model(model, dataset, max_iters=10)
    return fitted, metrics, dataset


# ---------- keypoint detection ----------


def test_video_reader_yields_correct_shapes(test_video):
    import cv2
    from monocular_demos.utils import video_reader

    cap = cv2.VideoCapture(test_video)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    vid, n_frames = video_reader(test_video, batch_size=4)
    assert n_frames == expected_frames
    frames_seen = 0
    for batch in vid:
        assert batch.shape[1:] == (expected_h, expected_w, 3)
        frames_seen += batch.shape[0]
    assert frames_seen == n_frames


def test_metrabs_detection_returns_correct_shapes(test_video, detected_keypoints):
    import cv2

    cap = cv2.VideoCapture(test_video)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    kp3d, kp2d, confs = detected_keypoints
    assert kp3d.shape == (expected_frames, N_JOINTS, 3)
    assert kp2d.shape == (expected_frames, N_JOINTS, 2)
    assert confs.shape == (expected_frames, N_JOINTS)


# ---------- biomechanical fitting ----------


def test_biomechanical_fitting_on_detected_keypoints(fitted_model):
    fitted, metrics, dataset = fitted_model
    assert callable(fitted)
    assert "total" in metrics
    assert len(dataset) == 1


# ---------- overlay ----------


def test_render_overlay_on_detected_and_fitted_data(test_video, fitted_model):
    import cv2
    import tempfile
    import shutil
    from monocular_demos.biomechanics_mjx.visualize import render_overlay

    fitted, _, dataset = fitted_model

    cap = cv2.VideoCapture(test_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # copy the video to a temp path so the overlay output doesn't land in the repo
    fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    shutil.copy2(test_video, tmp_video)
    output_path = tmp_video.replace(".mp4", "_overlay.mp4")

    fit_timestamps = dataset.get_all_timestamps(0)
    (state, _, _), (qpos, _, _), _ = fitted(
        fit_timestamps,
        trajectory_selection=0,
        steps=0,
        skip_action=True,
        fast_inference=True,
        check_constraints=False,
    )

    try:
        output = render_overlay(
            fit_timestamps=np.array(fit_timestamps),
            qpos=np.array(qpos),
            body_scale=np.array(fitted.body_scale),
            video_path=tmp_video,
            width=width,
            height=height,
            progress=lambda *a, **kw: None,
            max_frames=OVERLAY_MAX_FRAMES,
        )
        assert output is not None
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0
    finally:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        if os.path.exists(output_path):
            os.remove(output_path)
