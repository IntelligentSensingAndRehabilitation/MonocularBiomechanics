import mujoco
import numpy as np
from typing import List, Dict
from jaxtyping import Float, Array
from tqdm import trange, tqdm
import shutil
import subprocess
import cv2
from monocular_demos.biomechanics_mjx.forward_kinematics import (
    ForwardKinematics,
    scale_model,
)
from monocular_demos.dataset import get_samsung_calibration
from monocular_demos.biomechanics_mjx.monocular_trajectory import project_dynamic
import pyrender
from monocular_demos.camera import get_intrinsic, get_extrinsic, distort_3d
from stl import mesh  # install via pip install numpy-stl
import trimesh
import tempfile
import os



# use %env MUJOCO_GL=egl to avoid the need for a display
def render_trajectory(
    pose: Float[Array, "time n_pose"] | List[Dict],
    filename: str = None,
    mj_model: mujoco.MjModel = None,
    xml_path: str = None,
    body_scale: Float[Array, "nscale 1"] | None = None,
    site_offsets: np.array = None,
    margin: float | None = None,
    heel_vertical_offset: float | None = None,
    height: int = 480,
    width: int = 240,
    fps: int = 30,
    show_grfs: bool = False,
    actuators: Float[Array, "time n_actuators"] | None = None,
    azimuth: float = 135,
    blank_background: None | int = None,
    shadow=True,
    hide_tendon=False,
):
    """
    Renders a trajectory of poses using mujoco.

    Args:
        pose: trajectory of poses (or list of mjData)
        filename: if specified, saves the video to this file
        xml_path: path to the xml file (optional)
        site_offsets: offsets for the sites
        height: height of the video
        width: width of the video
        fps: frames per second
        show_grfs: whether to show ground reaction forces
        azimuth: azimuth of the camera
        blank_background: if specified, sets the background to this color (255 is white, 0 is black)
        shadow: whether to show shadows

    Returns:
        if filename is None, returns the images as a list
    """

    from monocular_demos.biomechanics_mjx.forward_kinematics import (
        ForwardKinematics,
        offset_sites,
        scale_model,
        set_margin,
        shift_geom_vertically,
    )
    import numpy as np

    fk = ForwardKinematics(xml_path=xml_path)

    if mj_model is not None:
        print("Using provided mj_model")
        model = mj_model
    else:
        model = fk.model  # non-mjx model

        if margin is not None:
            model = set_margin(model, margin)

        if heel_vertical_offset is not None:
            heel_geom_names = [
                "l_foot_col1",
                "l_foot_col3",
                "r_foot_col1",
                "r_foot_col3",
            ]
            heel_idx = np.array([fk.geom_names.index(g) for g in heel_geom_names])
            model = shift_geom_vertically(model, heel_idx, heel_vertical_offset)

    if body_scale is not None:
        scale = 1 + fk.build_default_scale_mixer() @ body_scale
        model = scale_model(model, scale)

    if site_offsets is not None:
        model = offset_sites(model, site_offsets)

    data = mujoco.MjData(model)

    scene_option = mujoco.MjvOption()
    # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    if show_grfs:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        if actuators is not None:
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

    if blank_background is not None:
        model.tex_data = np.zeros_like(model.tex_data) + blank_background

    if not shadow:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
        scene_option.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    if hide_tendon:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False

    # remove weird overlay
    geom_1_indices = np.where(model.geom_group == 1)
    model.geom_rgba[geom_1_indices, 3] = 0

    camera = mujoco.MjvCamera()
    camera.distance = 3
    camera.azimuth = azimuth
    camera.elevation = -20

    renderer = mujoco.Renderer(model, height=height, width=width)

    images = []
    for i in trange(len(pose)):
        if isinstance(pose, list):
            # when we are passed a list of mj_data just use them
            data = pose[i]

            # relight as otherwise super dark
            mujoco.mj_camlight(model, data)

        else:
            data.qpos = pose[i]
            mujoco.mj_forward(model, data)

        if i == 0:
            camera.lookat = data.xpos[1]
        else:
            camera.lookat = camera.lookat * 0.7 + data.xpos[1] * 0.3

        if actuators is not None:
            data.ctrl = data.ctrl * 0.0
            data.act = actuators[i]

        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        images.append(renderer.render())

    if filename is not None:
        import cv2
        import numpy as np

        cap = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        for i in trange(len(images)):
            # fix the color channels
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cap.write(img)

        cap.release()

    else:
        return images

def get_composed_meshes(model, data):
    """
    Compute and return a combined mesh by processing a MuJoCo model at a specific posture provided in data.

    Parameters:
    - model: the MuJoCo model containing geometry and mesh data
    - data: the data object containing geometry position and rotation data

    Returns:
    - combined: a single mesh object representing the combination of all individual meshes
    """
    t_mesh_L = []
    id_geom_with_mesh = np.where(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)[0]
    id_mesh = model.geom_dataid[id_geom_with_mesh]

    for id_g, rel_i in zip(id_geom_with_mesh, id_mesh):
        offset = data.geom_xpos[id_g].copy()
        rot = data.geom_xmat[id_g].copy()

        vert_r_t_list = []
        face_r_t_list = []

        for iv in np.arange(
            model.mesh_vertadr[rel_i],
            model.mesh_vertadr[rel_i] + model.mesh_vertnum[rel_i],
        ):
            # rotation = Rotation.from_matrix(rot.reshape(3, 3))
            # vert_r_t = offset[:] + rotation.apply(model.mesh_vert[iv, :])
            vert_r_t = offset[:] + (rot.reshape(3, 3) @ model.mesh_vert[iv, :].T).T
            vert_r_t_list.append(vert_r_t)

        face_r_t = model.mesh_face[
            model.mesh_faceadr[rel_i] : model.mesh_faceadr[rel_i] + model.mesh_facenum[rel_i],
            :,
        ]
        face_r_t_list.append(face_r_t)

        vertices = np.array(vert_r_t_list)
        faces = np.array(face_r_t)

        t_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                t_mesh.vectors[i][j] = vertices[f[j], :]
        t_mesh_L.append(t_mesh)

    combined = mesh.Mesh(np.concatenate([m.data for m in t_mesh_L]))
    vertices = combined.vectors.reshape((-1, 3))
    faces = np.arange(len(vertices)).reshape((-1, 3))

    return vertices, faces

def get_overlay_monocular(
    camera_params,
    rvec,
    scaled_model,
    poses,
    pose_frame_idx,
    camera_visible,
    width: int = 1080,
    height: int = 1920,
    downsample: int = 1,
):
    """
    Create an overlay function that renders the biomechanics model on top of the camera image.

    Args:
        camera_params: the camera parameters
        scaled_model: the scaled model
        poses: the poses
        pose_frame_idx: the pose frame index
        downsample: the downsample factor
    """

    from monocular_demos.camera import get_extrinsic_dynamic

    K = get_intrinsic(camera_params, 0)
    camera_pose_timeseries = np.array(get_extrinsic_dynamic(camera_params, 0, rvec))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Camera setup (assuming you have fx, fy, cx, cy, and camera pose)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=20)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.3,  # Increased metallic factor
        roughnessFactor=0.4,  # Adding roughness factor
        baseColorFactor=(0.8, 0.5, 0.5, 1.0),  # A neutral color closer to white
    )

    # Rendering
    r = pyrender.OffscreenRenderer(viewport_width=width // downsample, viewport_height=height // downsample)

    data = mujoco.MjData(scaled_model)

    def overlay(frame, idx):

        if idx in pose_frame_idx:
            idx = np.where(idx == pose_frame_idx)[0][0]
        else:
            return frame

        if not camera_visible[idx]:
            return frame

        # get frame-specific camera pose
        camera_pose = camera_pose_timeseries[idx]
        camera_pose[[1, 2]] *= -1
        camera_pose[:3, -1] /= 1000.0
        camera_pose = np.linalg.inv(camera_pose)

        data.qpos = poses[idx]
        mujoco.mj_forward(scaled_model, data)
        vertices, faces = get_composed_meshes(scaled_model, data)

        def correct_vertices(vertices):
            vertices = distort_3d(camera_params, 0, vertices * 1000)
            vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1], 1))], axis=-1)
            extri = np.linalg.inv(get_extrinsic(camera_params, 0))
            vertices = (extri @ vertices[..., None])[..., 0]
            vertices = vertices[:, :3] / vertices[:, 3, None]
            vertices = vertices / 1000.0
            return vertices

        vertices = correct_vertices(vertices)
        mesh = trimesh.Trimesh(vertices, faces)
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)  # , smooth=True, wireframe=True)

        # Create a scene
        scene = pyrender.Scene(ambient_light=(0.9, 0.9, 0.9))

        # Adding directional light
        directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(directional_light, pose=camera_pose)

        # Add the mesh to the scene
        scene.add(render_mesh)

        scene.add(camera, pose=camera_pose)

        color, depth = r.render(scene)

        if frame is None:
            return color

        else:
            mask = depth > 0
            frame[mask] = color[mask]
            return frame

    return overlay

def render_overlay(
        fit_timestamps,
        qpos,
        body_scale,
        video_path,
        width,
        height,
        progress,
        max_frames=None,
    ):

    camera_params = get_samsung_calibration()

    fk = ForwardKinematics()
    scale = 1 + fk.build_default_scale_mixer() @ body_scale
    scaled_model = scale_model(fk.model, scale)
    rvec = np.zeros((len(fit_timestamps), 3))

    fit_inds = np.arange(len(fit_timestamps))
    confs = np.ones((len(fit_timestamps)))

    mesh_overlay = get_overlay_monocular(camera_params, rvec, scaled_model, qpos, fit_inds, confs, width, height)

    def overlay(image, idx):
        progress(idx / (max_frames if max_frames is not None else rvec.shape[0]), desc="Rendering Overlay...")
        return mesh_overlay(image, idx)

    fd, out_file_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    video_overlay(video_path, out_file_name, overlay, downsample=1, compress=True, max_frames=max_frames)

    # move tempfile to final location
    shutil.move(out_file_name, video_path.split('.')[0] + '_overlay.mp4')
    return video_path.split('.')[0] + '_overlay.mp4'


def video_overlay(
    video,
    output_name,
    callback,
    downsample=4,
    codec="MP4V",
    compress=True,
    bitrate="5M",
    max_frames=None,
):
    """Process a video and create overlay image

    Args:
        video (str): filename for source
        output_name (str): output filename
        callback (fn(im, idx) -> im): method to overlay frame
    """

    cap = cv2.VideoCapture(video)

    # get info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # configure output
    output_size = (int(w / downsample), int(h / downsample))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_name, fourcc, fps, output_size)

    if max_frames:
        total_frames = max_frames

    for idx in tqdm(range(total_frames)):

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # process image in RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = callback(frame, idx)

        # move back to BGR format and write to movie
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        out_frame = cv2.resize(out_frame, output_size)
        out.write(out_frame)

    out.release()
    cap.release()

    if compress:
        fd, temp = tempfile.mkstemp(suffix=".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", output_name, "-c:v", "libx264", "-b:v", bitrate, temp])
        os.close(fd)
        shutil.move(temp, output_name)


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    plot_radius = min([plot_radius, 1000])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def render_trajectory_keypoints(
    k3d: Float[Array, "time n_keypoints 3"],
    k3d_gt: Float[Array, "time n_keypoints 3"] = None,
    gt_confidence: Float[Array, "time n_keypoints"] = None,
    confidence_threshold=0.1,
    filename: str = None,
    fps: int = 30,
):
    """
    Create a 3D scatter plot time series video from a numpy array.

    Parameters:
    k3d (np.ndarray): 3D keypoints
    k3d_gt (np.ndarray): An optional second set of 3D keypoints (displayed in black).
    filename (str): The name of the output video file.
    fps (int): Frames per second for the video.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    if k3d_gt is not None:
        assert (
            k3d.shape[0] == k3d_gt.shape[0]
        ), "The number of frames must be the same for both sets of keypoints."
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set up the writer
    metadata = dict(
        title="3D Scatter Plot Time Series",
        artist="Matplotlib",
        comment="Scatter plot animation",
    )
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(fig, filename, dpi=100):
        for t in tqdm(range(k3d.shape[0])):
            ax.cla()  # Clear the plot
            ax.scatter(k3d[t, :, 0], k3d[t, :, 1], k3d[t, :, 2], c="blue")
            if k3d_gt is not None:
                # only plot confident points
                if gt_confidence is not None:
                    confident_points = gt_confidence[t] > confidence_threshold
                    ax.scatter(
                        k3d_gt[t, confident_points, 0],
                        k3d_gt[t, confident_points, 1],
                        k3d_gt[t, confident_points, 2],
                        c="black",
                    )
                else:
                    ax.scatter(
                        k3d_gt[t, :, 0], k3d_gt[t, :, 1], k3d_gt[t, :, 2], c="black"
                    )
                # my_max = k3d.max(axis=1).max(axis=0)
                # ax.scatter(my_max[0],my_max[1],my_max[2],c='r')
                # my_min = k3d.min(axis=1).min(axis=0)
                # ax.scatter(my_min[0],my_min[1],my_min[2])

            set_axes_equal(ax)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Time: {t}")
            if t == 0:
                fig.legend(["Predicted", "Ground Truth"], loc="upper right")

            writer.grab_frame()
    plt.close(fig)


def jupyter_embed_video(video_filename: str):

    from IPython.display import HTML
    import subprocess
    import tempfile
    from base64 import b64encode
    import os

    # get temporary mp4 output
    fid, fn = tempfile.mkstemp(suffix=".mp4")
    # close fid
    os.close(fid)

    subprocess.run(
        ["ffmpeg", "-y", "-i", video_filename, "-hide_banner", "-loglevel", "error", fn]
    )

    video = open(fn, "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = f'<video controls src="data:video/mp4;base64,{video_encoded}">'

    os.remove(fn)

    return HTML(video_tag)
