"""
Tools for a camera model implementation in Jax
"""

import numpy as np
import jax
from jax import numpy as jnp
from jax import vmap, jit
from jax.tree_util import tree_map
from jaxlie import SO3, SE3
from functools import partial


def get_intrinsic(camera_params, i):
    mtx = jnp.take(camera_params["mtx"], i, axis=0)
    return jnp.array([[mtx[0] * 1000.0, 0, mtx[2] * 1000.0], [0, mtx[1] * 1000.0, mtx[3] * 1000.0], [0, 0, 1]])


def get_extrinsic(camera_params, i):
    rvec = jnp.take(camera_params["rvec"], i, axis=0)
    tvec = jnp.take(camera_params["tvec"], i, axis=0) * 1000.0
    rot = SO3.exp(rvec)
    return SE3.from_rotation_and_translation(rot, tvec).as_matrix()


def get_projection(camera_params, i):
    intri = get_intrinsic(camera_params, i)
    extri = get_extrinsic(camera_params, i)
    return intri @ extri[:3]


def project(camera_params, i, points):
    # make sure to use homogeneous coordinates
    if points.shape[-1] == 3:
        points = jnp.concatenate([points, jnp.ones((*points.shape[:-1], 1))], axis=-1)

    # last dimension ensures broadcasting works
    proj = get_projection(camera_params, i) @ points[..., None]
    proj = proj[..., 0]

    # remove affine dimension and get u,v coordinates
    return proj[..., :-1] / proj[..., -1:]


@jit
def project_distortion(camera_params, i, points):
    intri = get_intrinsic(camera_params, i)
    extri = get_extrinsic(camera_params, i)

    # make sure to use homogeneous coordinates
    if points.shape[-1] == 3:
        points = jnp.concatenate([points, jnp.ones((*points.shape[:-1], 1))], axis=-1)

    # transform the points into the camera perspective
    # last dimension ensures broadcasting works
    transformed = (extri @ points[..., None])[..., 0]

    distance = jnp.abs(transformed[..., 2])

    xp = transformed[..., 0] / jnp.where(distance < 1e2, 1e6, transformed[..., 2])
    yp = transformed[..., 1] / jnp.where(distance < 1e2, 1e6, transformed[..., 2])
    r2 = xp**2 + yp**2
    r2 = jnp.where(jnp.isnan(r2), 0, r2)

    dist = jnp.take(camera_params["dist"], i, axis=0)
    gamma = 1.0 + dist[0] * r2  # + dist[1] * r2**2 + dist[4] * r2**3

    # in the case of negative points, shrink them very close to center of screen
    # this is to make sure that calibration can't "see" through the back of the
    # camera
    negative = transformed[..., 2] < 0
    negative_scale = jnp.where(negative, 1e-3, 1)
    gamma = gamma * negative_scale

    xpp = gamma * xp  # + 2*dist[2]*xp*yp + dist[3] * (r2 + 2 * xp**2)
    ypp = gamma * yp  # + dist[2]*(r2 + 2*yp**2) + 2*dist[3]*xp*yp

    points = jnp.stack([xpp, ypp, jnp.ones(xpp.shape)], axis=-1)
    proj = (intri @ points[..., None])[..., 0]
    # remove affine dimension and get u,v coordinates

    return jnp.where(jnp.abs(proj[..., -1:]) < 1e-6, 0, proj[..., :-1] / proj[..., -1:])

@jit
def distort_3d(camera_params, i, points):
    """
    Transform points into camera frame and distort them by camera model.

    Typically points are distorted and projected, but instead this works out
    the 3D coordinates that would project with an ideal pinhole camera to
    the same coordinates ultimate image coordinates. These points don't
    correspond to anything true in the world, but are useful for rendering
    3D models overlaid on the true camera model (e.g. SMPL models). Since
    equivalent 3D points exist along a ray, this (arbitrarily) preserves
    the Z coordinate.

    Args:
        camera_params: dictionary of calibrated camera parameters
        i: which camera to use
        points: (*, N, 3) points in 3D world coordinates

    Returns:
        (*, N, 3) distorted points in camera coordinates
    """

    extri = get_extrinsic(camera_params, i)

    # make sure to use homogeneous coordinates
    if points.shape[-1] == 3:
        points = jnp.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)

    # transform the points into the camera perspective
    # last dimension ensures broadcasting works
    transformed = (extri @ points[..., None])[..., 0]

    z = transformed[..., 2]
    xp = transformed[..., 0] / z
    yp = transformed[..., 1] / z
    r2 = xp**2 + yp**2

    dist = camera_params["dist"][i]
    gamma = 1.0 + dist[0] * r2 + dist[1] * r2**2 + dist[4] * r2**3

    xpp = gamma * xp + 2 * dist[2] * xp * yp + dist[3] * (r2 + 2 * xp**2)
    ypp = gamma * yp + dist[2] * (r2 + 2 * yp**2) + 2 * dist[3] * xp * yp

    return jnp.stack([xpp * z, ypp * z, z], axis=-1)

def get_extrinsic_dynamic(camera_params, i, rvec):
    tvec = jnp.take(camera_params["tvec"], i, axis=0) * 1000.0

    def _get_extrinsic(_rvec):
        rot = SO3.exp(-1 * _rvec)
        return SE3.from_rotation_and_translation(rot, tvec).as_matrix()

    return jax.vmap(_get_extrinsic)(rvec)