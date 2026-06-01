import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def test_visualize_importable():
    from monocular_demos.biomechanics_mjx.visualize import render_trajectory, jupyter_embed_video
    assert callable(render_trajectory)
    assert callable(jupyter_embed_video)


def test_get_composed_meshes_returns_vertices_and_faces():
    import mujoco
    from monocular_demos.biomechanics_mjx.visualize import get_composed_meshes
    from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics

    fk = ForwardKinematics()
    model = fk.model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    vertices, faces = get_composed_meshes(model, data)

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3


def test_get_overlay_monocular_returns_callable():
    from monocular_demos.biomechanics_mjx.visualize import get_overlay_monocular
    from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinematics
    from monocular_demos.dataset import get_samsung_calibration

    fk = ForwardKinematics()
    model = fk.model
    n_frames = 3
    poses = np.zeros((n_frames, model.nq))
    pose_frame_idx = np.arange(n_frames)
    camera_visible = np.ones(n_frames, dtype=bool)
    camera_params = get_samsung_calibration()
    rvec = np.zeros((n_frames, 3))

    overlay_fn = get_overlay_monocular(camera_params, rvec, model, poses, pose_frame_idx, camera_visible)
    assert callable(overlay_fn)


def test_video_overlay_produces_output_file():
    import cv2
    import tempfile
    from monocular_demos.biomechanics_mjx.visualize import video_overlay

    fd, input_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        writer = cv2.VideoWriter(input_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
        for _ in range(5):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()

        video_overlay(input_path, output_path, callback=lambda frame, idx: frame, compress=False)

        assert os.path.getsize(output_path) > 0
    finally:
        os.remove(input_path)
        os.remove(output_path)
