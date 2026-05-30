import functools
import inspect
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union

import equinox as eqx
import jax
import mujoco
import optax
from jax import numpy as jnp
from jaxlie import SE3, SO3
from jaxtyping import Array, Float, Integer, PRNGKeyArray
from monocular_demos.camera import get_extrinsic, project_distortion
from monocular_demos.dataset import (
    KeypointDataset,
    MonocularDataset,
    level_floor_from_dataset,
)
from monocular_demos.implicit_representation import (
    CubicSplineTrajectory,
    ImplicitTrajectory,
    calculate_encoding_length,
)
from mujoco import mjx
from tqdm import trange

from .forward_kinematics import (
    ForwardKinematicsModelDescriptor,
    ForwardKinetics,
    angular_velocity_to_quaternion_derivative,
    normalize_velocity_from_quaternions,
)


class KineticsWrapper(eqx.Module):
    trajectory: ImplicitTrajectory | CubicSplineTrajectory
    body_scale: jnp.array
    scale_mixer: jnp.array
    site_offsets: jnp.array
    lower_limit: jnp.array
    upper_limit: jnp.array
    apply_limits: jnp.array
    njoints: int = eqx.field(static=True)  # number of qpos joints
    nvjoints: int = eqx.field(static=True)  # number of vel joints (can be less than qpos if quaternions)
    nactions: int = eqx.field(static=True)
    nextra: int = eqx.field(static=True)
    forward: ForwardKinematicsModelDescriptor
    jit_forward: Callable = eqx.field(static=True)  # ForwardKinetics
    freeroot: bool = eqx.field(static=True)

    def __init__(
        self,
        kinetic_model: ForwardKinetics,
        trajectory=None,
        trajectory_type="implicit",
        scale_mixer=None,
        num_trajectories=1,
        encoding_length=None,
        cubic_sequence_length=None,
        max_time=60,
        min_freq=80,
        key=None,
        action=True,  # set to false for kinematic fits
        nextra=0,  # extra outpts from the implicit
        **kwargs,
    ):
        self.njoints = kinetic_model.mjx_model.nq
        self.nvjoints = kinetic_model.mjx_model.nv
        self.nactions = kinetic_model.mjx_model.nu
        nbody = kinetic_model.mjx_model.nbody
        self.nextra = nextra
        self.freeroot = True if kinetic_model.mjx_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else False

        max_time += 1.0  # add an additional second of representation space to cover the last frame future predictions

        if action:
            dims = self.njoints + self.nactions
        else:
            # for kinematic fits
            dims = self.njoints
            self.nactions = 0

        if key is None:
            key = jax.random.PRNGKey(0)

        # adds extra output dims
        dims += nextra
        if trajectory is None:

            trajectory_keys = jax.random.split(key, num_trajectories)
            if trajectory_type.lower() == "implicit":

                if encoding_length is None:
                    encoding_length = calculate_encoding_length(max_time, min_freq=min_freq)

                self.trajectory = [
                    ImplicitTrajectory(joints=dims, spatial_dims=1, encoding_length=encoding_length, max_time=max_time, **kwargs, key=key)
                    for key in trajectory_keys
                ]
            elif trajectory_type.lower() == "cubic":

                if cubic_sequence_length is None:
                    cubic_sequence_length = int(max_time * min_freq)
                self.trajectory = [
                    CubicSplineTrajectory(sequence_length=cubic_sequence_length, joints=dims, spatial_dims=1, max_time=max_time, key=key)
                    for key in trajectory_keys
                ]

        else:
            self.trajectory = trajectory

        if not isinstance(self.trajectory, list):
            self.trajectory = [self.trajectory]

        # make the trajectory something we can index it to while jitting
        def f(*args):
            return jnp.stack(list(args), axis=0)

        self.trajectory = jax.tree_util.tree_map(f, *self.trajectory)
        self.scale_mixer = scale_mixer if scale_mixer is not None else jnp.ones((nbody, 1))

        dof_qpos_jntid = jnp.array(kinetic_model.dof_qpos_jntid)
        if action:
            self.lower_limit = jnp.concatenate(
                [kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 0], kinetic_model.mjx_model.actuator_ctrlrange[:, 0], -jnp.ones(nextra)]
            )
            self.upper_limit = jnp.concatenate(
                [kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 1], kinetic_model.mjx_model.actuator_ctrlrange[:, 1], jnp.ones(nextra)]
            )
            self.apply_limits = (
                jnp.concatenate(
                    [
                        kinetic_model.mjx_model.jnt_limited[dof_qpos_jntid],
                        kinetic_model.mjx_model.actuator_ctrllimited,
                        jnp.zeros(nextra),
                    ]
                )
                >= 1
            )
        else:
            # for kinematic fits
            self.lower_limit = jnp.concatenate([kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 0], -jnp.ones(nextra)])
            self.upper_limit = jnp.concatenate([kinetic_model.mjx_model.jnt_range[dof_qpos_jntid, 1], jnp.ones(nextra)])
            self.apply_limits = jnp.concatenate([kinetic_model.mjx_model.jnt_limited[dof_qpos_jntid], jnp.zeros(nextra)]) >= 1
        self.body_scale = jnp.zeros((self.scale_mixer.shape[1], 1))
        self.site_offsets = jnp.zeros((kinetic_model.mjx_model.nsite, 3))

        self.forward = kinetic_model.get_descriptor()
        # NOTE: this throws a warning.
        # UserWarning: A JAX array is being set as static! This can result in unexpected behavior and is usually a mistake to do.
        # but doing the compilation below is quite slow
        self.jit_forward = eqx.filter_jit(kinetic_model)

    # @functools.partial(eqx.filter_jit, static_argnames=("check_constraints", "steps", "freeze_scale"))
    def __call__(
        self,
        timepoint,
        trajectory_selection=0,
        trajectory_only=False,
        skip_action=False,
        skip_vel=False,
        steps=1,
        dt=1.0 / 30.0,
        fast_inference=False,
        freeze_scale=False,
        margin=None,
        action_repeats=1,
        check_constraints=True,
    ):
        scale = 1 + jax.lax.stop_gradient(self.scale_mixer) @ self.body_scale

        scale = jax.lax.cond(freeze_scale, lambda: jax.lax.stop_gradient(scale), lambda: scale)
        site_offsets = jax.lax.cond(freeze_scale, lambda: jax.lax.stop_gradient(self.site_offsets), lambda: self.site_offsets)

        # NOTE: the dt passed here should be the frame timestep divided by the frame interpolation
        # here, we additionally rescale it by the number of action repeats
        model_timestep = None if skip_action else dt / action_repeats
            
        trajectory = jax.tree_util.tree_map(lambda x: jnp.take(x, trajectory_selection, axis=0), self.trajectory)

        if fast_inference:

            def vectorize(f: Callable):
                # vectorize using a scan to avoid the compilation time of vmap
                def scan_fn(carry, x):
                    return (carry + 1, f(x))

                return lambda x: jax.lax.scan(scan_fn, 0, x)[1]

        else:
            vectorize = jax.vmap

        def scale_limits(x):
            # broadcast lower_limit and upper limit to be the same shape and
            # last dimension as the pose

            lower_limit = jax.lax.stop_gradient(jnp.broadcast_to(self.lower_limit, x.shape))
            upper_limit = jax.lax.stop_gradient(jnp.broadcast_to(self.upper_limit, x.shape))

            range = (upper_limit - lower_limit) / 2.0
            # for joints with no range in the model, prevent gradients having divide by zero
            range = jnp.clip(range, 0.01, None)
            middle = (upper_limit + lower_limit) / 2.0

            # apply the bounded limits to all of the joints but ignore the global pose
            # this includes allowing the global pose to wrap around
            scaled = jnp.tanh(x) * range + middle
            x = jnp.where(self.apply_limits, scaled, x)
            return x

        def get_pose_and_action(t):
            pose_action_extra = trajectory(t)[..., 0]
            pose_action_extra = scale_limits(pose_action_extra)
            pose = pose_action_extra[..., : self.njoints]
            action = pose_action_extra[..., self.njoints : (self.njoints + self.nactions)]
            extra = pose_action_extra[..., (self.njoints + self.nactions) :]
            return pose, action, extra

        def process_single_time(t):
            pose, action, extra = get_pose_and_action(t)

            if skip_vel:
                vel = jnp.zeros((self.nvjoints))
            else:
                # analyztically compute joint velocities
                vel = jax.jacfwd(lambda x: scale_limits(trajectory(x)[:, 0]))(t)
                raw_vel = vel[..., : self.njoints]

                # for freeroot, this changes from derivative w.r.t. qpos to the qvel with one less element
                vel = normalize_velocity_from_quaternions(pose, raw_vel, self.freeroot)

            if trajectory_only:
                return pose, vel, action, extra

            if skip_action:
                return (
                    self.jit_forward(
                        joint_angles=pose,
                        joint_velocities=vel,
                        scale=scale,
                        site_offsets=site_offsets,
                        margin=margin,
                        check_constraints=check_constraints,
                        timestep=model_timestep,
                    ),
                    (pose, vel, action),
                    extra,
                )

            def scan_function(carry, i):
                pose, action, extra = get_pose_and_action(t + i * dt)
                vel = jax.jacfwd(lambda x: scale_limits(trajectory(x)[:, 0]))(t + i * dt)[..., : self.njoints]
                vel = normalize_velocity_from_quaternions(pose, vel, self.freeroot)

                return carry, (pose, vel, action)

            _, (future_pose, future_vel, action) = jax.lax.scan(scan_function, (), jnp.arange(steps + 1))
            action = action[:-1]

            rollout = lambda x, y, z: self.jit_forward(
                joint_angles=x,
                joint_velocities=y,
                action=z,
                scale=scale,
                site_offsets=site_offsets,
                margin=margin,
                action_repeats=action_repeats,
                check_constraints=check_constraints,
                timestep=model_timestep,
            )
            return rollout(pose, vel, action), (future_pose, future_vel, action), extra

        return vectorize(process_single_time)(timepoint)



def huber(x, delta=5.0):
    """Huber loss."""
    x = jnp.where(jnp.abs(x) < delta, 0.5 * x**2, delta * (jnp.abs(x) - 0.5 * delta))
    return x

def get_default_wrapper():
    """
    Convenience function to get a model for the 3D MOVI keypoints.

    Returns:
        A tuple of the model and loss function.
    """

    from monocular_demos.biomechanics_mjx.forward_kinematics import ForwardKinetics,movi_joint_names

    # default forward kinematic model
    fk = ForwardKinetics(site_reorder=movi_joint_names)

    # default scaler mixer
    scale_mixer = fk.build_default_scale_mixer()

    # use the implicit function wrapper
    fkw = KineticsWrapper(fk, scale_mixer=scale_mixer, key=jax.random.PRNGKey(0))

    return fkw

def get_custom_wrapper(fk: ForwardKinetics, dataset: MonocularDataset, dist=1.5, **kwargs) -> KineticsWrapper:
    """This will buiild an implicit trajectory with custom initializations computed from the dataset."""
    import numpy as np
    from jaxlie import SO3

    min_freq = kwargs.get("min_freq", 80)  # safely handle default

    dims = fk.mjx_model.nq + kwargs["nextra"]
    if "encoding_length" not in kwargs:
        encoding_length = calculate_encoding_length(dataset.max_time, min_freq=min_freq)
    else:
        encoding_length = kwargs["encoding_length"]

    # calculate custom output layer bias
    custom_biases = []
    if dataset.phone_attitude is not None:
        rvec_means = [
            np.median(dataset.get_all_attitude(i)[1], axis=0)
            for i in range(len(dataset))
        ]

        for rvec in rvec_means:
            custom_bias = np.full((dims,), np.nan)

            # Set rotation bias from median rotation vector
            custom_bias[-3:] = rvec

            # Convert rotation vector to rotation matrix and extract direction
            rmat = SO3.exp(rvec).as_matrix()
            z_axis = rmat[:, 2]

            # Project direction onto ground plane (XY plane)
            ground_normal = np.array([0, 0, 1])
            vproj = z_axis - np.dot(z_axis, ground_normal) * ground_normal
            vproj /= np.linalg.norm(vproj)

            # Estimate 2D position in XY plane, assuming person faces away from camera
            x_pos = vproj[0] * dist
            y_pos = vproj[1] * dist

            # Assign position and facing direction depending on freeroot/not
            if fk.joint_names[0].startswith("root"):
                custom_bias[3:5] = 0.7071
                custom_bias[:3] = np.array([x_pos, y_pos, -1.5])
            else:
                custom_bias[:3] = np.array([x_pos, -y_pos, -1.5])

            custom_biases.append(custom_bias)

    else:
        # For staic fits generally the camera is not at origin so just get the freeroot right
        for _ in range(len(dataset)):
            custom_bias = np.full((dims,), np.nan)
            if fk.joint_names[0].startswith("root"):
                custom_bias[3:5] = 0.7071
            custom_biases.append(custom_bias)


    trajectory_keys = jax.random.split(jax.random.PRNGKey(0), len(dataset))
    custom_implicits = [
        ImplicitTrajectory(
            joints=dims, spatial_dims=1, max_time=dataset.max_time, encoding_length=encoding_length, custom_bias=custom_biases[i], key=key
        )
        for i, key in enumerate(trajectory_keys)
    ]

    custom_trajectory = KineticsWrapper(
        fk,
        scale_mixer=fk.build_default_scale_mixer(),
        key=jax.random.PRNGKey(0),
        num_trajectories=len(dataset),
        max_time=dataset.max_time,
        action=False,
        trajectory=custom_implicits,
        **kwargs,
    )

    return custom_trajectory


# this could be deleted and just used from Kinetic trajectory. leaving it here for now in case I write a new LCR class.
def get_model(dataset: KeypointDataset | MonocularDataset, xml_path=None, joint_names=None, scale_mappings=None, **kwargs) -> KineticsWrapper:
    """
    Get a kinetics wrapper trajectory for training from a dataset.

    Args:
        dataset (KeypointDataset): The dataset.
        camera_params (Dict): The camera parameters.

    Returns:
        Tuple[KineticsWrapper, LearnedCameraReprojection]: The model.
    """

    if joint_names is None:
        import warnings

        warnings.warn("Using default joint names. This may not be correct for your dataset.")
        joint_names = (KeypointSet & {"xml_fname": xml_path.split("/")[-1]}).get_site_reorder()

    # default forward kinematic model
    fk = ForwardKinetics(xml_path=xml_path, site_reorder=joint_names)

    if scale_mappings is None:
        scale_mixer = fk.build_default_scale_mixer()
        print("Using default scale mixer.")
    else:
        scale_mixer = fk.build_custom_scale_mixer(scale_mappings)

    if dataset.phone_attitude is not None or fk.joint_names[0].startswith("root"):
        trajectory = get_custom_wrapper(fk, dataset, **kwargs)
    else:
        trajectory = KineticsWrapper(
            fk,
            scale_mixer=scale_mixer,
            key=jax.random.PRNGKey(0),
            num_trajectories=len(dataset),
            max_time=dataset.max_time,
            action=False,
            **kwargs,
        )

    model = trajectory

    return model


def project_dynamic(points: Float[Array, "time joints 3"], calibation: Dict, rvec):

    camera_params = {"mtx": calibation["mtx"], "dist": calibation["dist"]}  # these do not vary with time
    tvec = calibation["tvec"]  # this is the same for all times

    def project_time(_rvec, _tvec, points, _camera_params=camera_params):
        _camera_params["rvec"] = -1 * _rvec.reshape(1, 3)  # reshape for compatibiltiy with multi-camera code
        _camera_params["tvec"] = _tvec.reshape(1, 3)
        return project_distortion(_camera_params, 0, points)

    return jax.vmap(project_time, in_axes=(0, None, 0))(rvec, tvec, points)  # for future changing tvec, remove this none


def get_extrinsic_dynamic(camera_params, i, rvec: Float[Array, "time 3"]):
    tvec = jnp.take(camera_params["tvec"], i, axis=0) * 1000.0

    def _get_extrinsic(_rvec):
        rot = SO3.exp(-1 * _rvec)
        return SE3.from_rotation_and_translation(rot, tvec).as_matrix()

    return jax.vmap(_get_extrinsic)(rvec)


# if you add args to this function the dont forget to add them to the vmap
def loss(
    model: KineticsWrapper,
    x: Tuple[Integer, Float[Array, "camera_times"], Float[Array, "attitude_times"]],
    y: Tuple[
        Float[Array, "cameras camera_times keypoints 2"],
        Float[Array, "cameras camera_times keypoints 3"],
        Float[Array, "cameras camera_times keypoints"],
        Float[Array, "attitude_times 3"],
    ],
    lambda_reprojection: Array,
    camera_calibration: Dict,
    site_offset_regularization: float = 1e2,
    static: bool = False,
    center_3d: bool = True,
    quat_regularization: float = 100.0,
    ang_pred_regularization: float = 0.0,
    ang_pred_vel_scale: float = 10.0,
    dt: float = 1.0 / 30.0,
) -> Tuple[Float, Dict]:
    """This is written for a single trajectory however will work for a batch of trajectories as well."""

    # vmap over the data so this is applied to a single trial
    if isinstance(x[0], jnp.ndarray) and len(x[0].shape) == 1:
        # if there is a batch dimension, automatically vmap this function over it
        _loss = lambda x, y: loss(
            model=model,
            x=x,
            y=y,
            lambda_reprojection=lambda_reprojection,
            site_offset_regularization=site_offset_regularization,
            camera_calibration=camera_calibration,
            static=static,
            center_3d=center_3d,
            quat_regularization=quat_regularization,
            ang_pred_regularization=ang_pred_regularization,
            ang_pred_vel_scale=ang_pred_vel_scale,
            dt=dt,
        )
        _loss = jax.vmap(_loss, in_axes=(0, 0), out_axes=(0, 0))
        l, metrics = _loss(x, y)
        return jnp.sum(l), metrics
    elif isinstance(x[0], jnp.ndarray) and len(x[0].shape) > 1:
        raise ValueError("Cannot batch over more than one dimension in the input data.")

    # unpack data
    trajectory_id, video_timestamps, attitude_timestamps = x
    keypoints2d, keypoints3d, keypoints_confidences, attitude = y
    keypoints_confidences = keypoints_confidences[0]

    # forward pass at video times
    (state, constraints, next_states), (ang, vel, action), rvec_video = model(
        video_timestamps,
        trajectory_selection=trajectory_id,
        skip_vel=False if ang_pred_regularization > 0 else True,
        skip_action=True,
        steps=0,
        margin=1e-3,
        action_repeats=0,
        dt=dt,
        check_constraints=False,
    )
    if static:
        lambda_attitude = 0.0
        l_attitude = 0.0
        T_nc = get_extrinsic(camera_calibration, 0)
        p_n = jnp.concatenate([state.site_xpos * 1000, jnp.ones((*state.site_xpos.shape[:-1], 1))], axis=-1)
        p_c_hat = (T_nc @ p_n[..., None])[..., :3, 0]
        model_keypoints_mm = p_c_hat

        # reprojection
        if lambda_reprojection > 0.0:
            projected_points = project_distortion(camera_calibration, 0, state.site_xpos * 1000)  # this is performing the extrinsic math twice
    else:
        rvec_video = SO3.exp(rvec_video).normalize().log()
        # attitude
        lambda_attitude = 1e0
        _, _, _, qnc_phone = model(
            attitude_timestamps,
            trajectory_selection=trajectory_id,
            skip_vel=True,
            skip_action=True,
            steps=0,
            margin=1e-3,
            action_repeats=0,
            dt=dt,
            trajectory_only=True,
        )
        # l_attitude = jnp.sum(jnp.square(attitude - qnc_phone))
        q_diff = (SO3.exp(attitude).inverse() @ SO3.exp(qnc_phone).normalize()).parameters()
        quat_angle = lambda q: 2 * jnp.arctan2(jnp.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2), jnp.abs(q[0]))
        l_attitude = jnp.mean(jax.vmap(quat_angle)(q_diff) * 180 / jnp.pi)

        # keypoint
        T_nc = get_extrinsic_dynamic(camera_calibration, 0, rvec_video)
        p_n = jnp.concatenate([state.site_xpos * 1000, jnp.ones((*state.site_xpos.shape[:-1], 1))], axis=-1)
        p_c_hat = jnp.einsum("tij,tbj->tbi", T_nc, p_n)[..., :3]
        model_keypoints_mm = p_c_hat

        # reprojection
        if lambda_reprojection > 0.0:
            projected_points = project_dynamic(state.site_xpos * 1000, camera_calibration, rvec_video)

    metrics = {}

    # region 3d loss
    lambda_3d = 1e0
    video_keypoints_mm = keypoints3d[0]
    # remove mean from keypoints
    if center_3d:
        model_keypoints_mm = model_keypoints_mm - jnp.mean(model_keypoints_mm, axis=1, keepdims=True)
        video_keypoints_mm = video_keypoints_mm - jnp.mean(video_keypoints_mm, axis=1, keepdims=True)
    dist_3d = huber(jnp.linalg.norm(model_keypoints_mm / 10 - video_keypoints_mm / 10, axis=-1) * keypoints_confidences, delta=10)
    norm = keypoints_confidences.sum() + 1e-7
    dist_3d = jnp.sum(dist_3d) / norm
    # endregion

    # site offset
    l_site_offset = jnp.sum(jnp.square(model.site_offsets))

    # reprojection loss
    if lambda_reprojection > 0.0:
        dist_2d = huber(jnp.linalg.norm(projected_points - keypoints2d[0], axis=-1) * keypoints_confidences, delta=50)
        dist_2d = jnp.nansum(dist_2d) / norm
    else:
        dist_2d = 0.0

    # Log raw metrics
    metrics = {
        "mean_3d_raw": dist_3d,
        "site_offset_raw": l_site_offset,
        "mean_reprojection_raw": dist_2d,
        "attitude_loss_raw": l_attitude,
    }

    # Calculate scaled metrics
    metrics["mean_3d_scaled"] = metrics["mean_3d_raw"] * lambda_3d
    metrics["site_offset_scaled"] = metrics["site_offset_raw"] * site_offset_regularization
    metrics["mean_reprojection_scaled"] = metrics["mean_reprojection_raw"] * lambda_reprojection
    metrics["attitude_loss_scaled"] = metrics["attitude_loss_raw"] * lambda_attitude

    # Total loss
    l = metrics["mean_3d_scaled"] + metrics["site_offset_scaled"] + metrics["attitude_loss_scaled"] + metrics["mean_reprojection_scaled"]

    # freeroot loss
    if model.freeroot:
        # NOTE: this only applies to the free root right now
        quat_error = 1 - jnp.linalg.norm(ang[..., 3:7], axis=-1)
        metrics["quat_error"] = jnp.mean(jnp.square(quat_error))
        l += metrics["quat_error"] * quat_regularization

    if ang_pred_regularization > 0:
        # use a simple future prediction to get velocity roughly smoothed out
        ang_dt, _, _, _ = model(video_timestamps + dt, trajectory_selection=trajectory_id, trajectory_only=True)

        # we are trying to use our estimates of the future angle to supervise the velocity -- but dont want
        # to use the noisy velocity estimates to influence the pose estimates so stop the gradient here
        ang_dt = jax.lax.stop_gradient(ang_dt)

        if model.freeroot:
            qdot = angular_velocity_to_quaternion_derivative(ang[..., 3:7], vel[..., 3:6])
            vel = jnp.concatenate([vel[..., :3], qdot, vel[..., 6:]], axis=-1)

        pred_ang = ang + vel * dt
        err = pred_ang - ang_dt
        err = err.at[:, :3].set(err[:, :3] * ang_pred_vel_scale)
        metrics["ang_pred"] = jnp.mean(jnp.square(err))
        l = l + metrics["ang_pred"] * ang_pred_regularization

    # make loss the first key in the dictionary by popping and building a new dictionary with the rest
    metrics = {"total": l, **metrics}

    return l, metrics


@eqx.filter_jit
def step(model, opt_state, data, loss_grad, optimizer, **kwargs):
    x, targets = data
    (val, metrics), grads = loss_grad(model, x=x, y=targets, **kwargs)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return val, model, opt_state, metrics


def fetch_data(i, dataset, N, model_device, robust_camera_weights=False):
    # fetch the next batch of data and move to the same device as the model
    idx = jnp.arange(N) + i * N
    if robust_camera_weights:
        data = dataset.get_with_camera_weights(idx)
    else:
        data = dataset[idx]
    data = jax.tree_util.tree_map(lambda x: jax.device_put(x, model_device), data)
    return data


def fit_model(
    model: KineticsWrapper,
    dataset: Tuple | MonocularDataset,
    lr_end_value: float = 1e-7,
    lr_init_value: float = 1e-4,
    warmup_steps: int = 0,
    transition_begin: int = 0,
    max_iters: int = 25000,
    site_offset_regularization: float = 1e2,
    lambda_reprojection: float = 1e-1,
    clip_by_global_norm: float = 0.1,
    optimizer_zero_nans: bool = False,
    center_3d: bool = True,
    step_callback: Callable | None = None,
    **kwargs,  # additional kwargs go to the loss function
):
    """
    Fits the model to the given keypoints.

    Parameters:
        model (Tuple[KineticsWrapper, LearnedCameraReprojection, KeypointLearnedDistribution]): The model.
        dataset (Tuple | KeypointDataset): The dataset.
        lr_end_value (float): The end value of the learning rate.
        lr_init_value (float): The initial value of the learning rate.
        max_iters (int): The maximum number of iterations.
        site_offset_regularization (float): The site offset regularization.
        calibration_learning_step (int): The calibration learning step.
        calibration_learning_rate (float): The calibration learning rate.
    """

    is_static = kwargs.pop("static", dataset.phone_attitude is None)

    # find the camera frame rate for the dataset
    if is_static:
        kwargs['dt'] = 1.0 / 29.0
    else:
        kwargs['dt'] = 1.0 / 30.0

    # work out the transition steps to make the desired schedule
    transition_steps = 10
    lr_decay_rate = (lr_end_value / lr_init_value) ** (1.0 / ((max_iters - warmup_steps) // transition_steps))
    learning_rate = optax.warmup_exponential_decay_schedule(
        init_value=0,
        peak_value=lr_init_value,
        end_value=lr_end_value,
        warmup_steps=warmup_steps,
        transition_begin=transition_begin,
        decay_rate=lr_decay_rate,
        transition_steps=transition_steps,
    )

    # create learning rates for different components
    components = [optax.adamw(learning_rate=learning_rate, b1=0.8, weight_decay=1e-5)]
    if optimizer_zero_nans:
        components.append(optax.zero_nans())
    if clip_by_global_norm > 0:
        components.append(optax.clip_by_global_norm(clip_by_global_norm))
    optimizer = optax.chain(*components)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_grad = eqx.filter_value_and_grad(
        functools.partial(
            loss,
            site_offset_regularization=site_offset_regularization,
            lambda_reprojection=lambda_reprojection,
            camera_calibration=dataset.camera_params,
            static=is_static,
            center_3d=center_3d,
            **kwargs
        ),
        has_aux=True,
    )

    N = len(dataset)
    model_device = list(model.scale_mixer.devices())[0]
    local_fetch_data = jax.jit(functools.partial(fetch_data, dataset=dataset, N=N, model_device=model_device))

    smoothed_metrics = None

    @jax.jit
    def update_metrics(new_metrics, old_metrics):
        new_metrics_reduced = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), new_metrics)

        # Function to check if the structures are the same
        def is_same_structure(x, y):
            if isinstance(x, dict) and isinstance(y, dict):
                if x.keys() != y.keys():
                    return False
                return all(is_same_structure(x[k], y[k]) for k in x)
            return type(x) == type(y)

        if old_metrics is None or not is_same_structure(new_metrics_reduced, old_metrics):
            return new_metrics_reduced
        else:
            return jax.tree_util.tree_map(lambda x, y: 0.9 * x + 0.1 * y, old_metrics, new_metrics_reduced)

    counter = trange(max_iters)
    for i in counter:
        data = local_fetch_data(jnp.array(i))
        val, model, opt_state, metrics = step(model, opt_state, data, loss_grad, optimizer)
        if i == 1000:
            if jnp.isnan(val):
                raise ValueError(f"Loss is NaN on iteration {i}.")

        smoothed_metrics = update_metrics(metrics, smoothed_metrics)

        if step_callback is not None:
            step_callback(i, model, dataset, {**metrics, "learning_rate": learning_rate(i)})

        if i > 0 and i % int(max_iters // 10) == 0:
            display_metrics = smoothed_metrics
            print(f"iter: {i} loss: {val}.")  # metrics: {metrics}")

        if i % 50 == 0:
            # now round all of them to three decimal places
            # metrics = {k: round(v, 3) for k, v in metrics.items()}
            display_metrics = {k: v.item() for k, v in smoothed_metrics.items()}

            # make this an OrderedDict and make sure "loss" is the first element, followed by "mean_reprojection" and then the rest
            ordered_display_metrics = OrderedDict(
                sorted(display_metrics.items(), key=lambda x: x[0] not in ["total", "mean_3d_raw", "site_offset_raw"])
            )
            counter.set_postfix(ordered_display_metrics)

            if i % int(max_iters // 10) == 0:
                print(display_metrics)

    return model, metrics
