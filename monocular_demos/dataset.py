import equinox as eqx
import numpy as np
import copy
from jax import numpy as jnp
import functools
import jax
from typing import List, Dict
from jaxtyping import Integer, Float, Array


class KeypointDataset(eqx.Module):
    """
    Wrapper for sourcing keypoints for fitting biomechanics

    This handles providing data from different number of timepoints and trials
    to allow efficient training.

    future_timestamps allows providing multiple timesteps into the future for a given
    timestamp, for performing mini-rollouts. For compatibility with code that does not
    support this, future_timestamps can be set to zero and we do not include a future
    time axis. For future_timestamps=0 this will have one timestep (the target with
    none in the future).

    This also supports stacking multiple datasets when performing trilevel optimization
    over multiple sessions. They must all then have the same max_length and number of
    trials for this to work.
    """

    timestamps: Float[Array, "times"]
    keypoints: Float[Array, "cameras times keypoints 2"]
    keypoint_confidence: Float[Array, "cameras times keypoints"]
    max_time: Float[Array, "1"]
    trial_lengths: jnp.array
    num_trials: int = eqx.field(static=True)
    sample_length: int = eqx.field(static=True)
    future_timestamps: int = eqx.field(static=True)
    keys: List[Dict] = eqx.field(static=True)

    def __init__(
        self,
        timestamps,
        keypoints,
        keypoint_confidence,
        sample_length=None,
        align_timestamps=True,
        max_length=None,
        future_timestamps=-1,
        keys=None,
    ):

        if sample_length is None:
            sample_length = max([len(t) for t in timestamps])

        if align_timestamps:
            timestamps = [t - t[0] for t in timestamps]

        if max_length is None:
            max_length = max([len(t) for t in timestamps])

        self.trial_lengths = jnp.array([min(len(t), max_length) for t in timestamps])
        self.future_timestamps = future_timestamps

        def padded_stack(y, time_axis=0):

            def trim(_y):
                if _y.shape[time_axis] > max_length:
                    _y = jnp.take(_y, jnp.arange(max_length), axis=time_axis)
                return _y

            y = [trim(_y) for _y in y]

            shape = list(y[0].shape)
            shape[time_axis] = max_length
            x = jnp.zeros((len(y), *shape))
            for i, t in enumerate(y):
                if time_axis == 0:
                    x = x.at[i, : t.shape[0]].set(t)
                elif time_axis == 1:
                    x = x.at[i, :, : t.shape[1]].set(t)
                else:
                    raise NotImplementedError("Only time axis 0 or 1 supported")
            return x

        self.timestamps = padded_stack(timestamps)
        self.keypoints = padded_stack(keypoints, 1)
        self.keypoint_confidence = padded_stack(keypoint_confidence, 1)

        self.max_time = jnp.array(max([t[-1] for t in timestamps]), dtype=jnp.float32)

        # static fields should not be jax types
        self.num_trials = len(keypoints)
        self.sample_length = int(sample_length)
        self.keys = keys

    def __len__(self):
        return jnp.array(self.num_trials * self.num_sessions)

    @eqx.filter_jit
    def __getitem__(self, idx, sample_length=None):

        sample_length = sample_length if sample_length is not None else self.sample_length

        if isinstance(idx, int):
            idx = jnp.array(idx)
            return self._internal_getitem(idx, sample_length)
        elif isinstance(idx, jnp.ndarray) and len(idx.shape) == 1:
            return jax.vmap(lambda _idx: self._internal_getitem(_idx, sample_length=sample_length))(idx)
        elif isinstance(idx, jnp.ndarray) and len(idx.shape) > 0:
            raise NotImplementedError("Higher dimensional indexing not supported")

    def _internal_getitem(self, idx, sample_length):
        # idx gets modulo'd to index into multiple flattened dimensions. first it indexes into trials within a session
        # then it wraps around and indexes sessions, and finally it wraps around and residual determines the start
        # offset

        total_trials = self.num_trials * self.num_sessions
        start_idx = idx // total_trials  # offset for start
        idx = (idx % total_trials).astype(int)
        session_idx = idx // self.num_trials
        trial_idx = idx % self.num_trials

        # handle where we have multiple sessions or a single session in a consistent manner
        if self.multisession:
            session_fetch = lambda x: jnp.take(x, session_idx, axis=0)
        else:
            session_fetch = lambda x: x

        session_trial_len = session_fetch(self.trial_lengths)
        session_timestamps = session_fetch(self.timestamps)
        session_keypoints = session_fetch(self.keypoints)
        session_keypoint_confidence = session_fetch(self.keypoint_confidence)

        trial_len = jnp.take(session_trial_len, trial_idx)
        timestamps = jnp.take(session_timestamps, trial_idx, axis=0)
        keypoints = jnp.take(session_keypoints, trial_idx, axis=0)
        keypoint_confidence = jnp.take(session_keypoint_confidence, trial_idx, axis=0)

        # compute deterministically interleaved samples
        N = jnp.ceil(trial_len / sample_length).astype(int)

        sample_idx = jnp.arange(sample_length) * N + start_idx
        sample_idx = (sample_idx % trial_len).astype(int)

        timestamps = jnp.take(timestamps, sample_idx, axis=0)
        if self.future_timestamps == -1:
            keypoints = jnp.take(keypoints, sample_idx, axis=1)
            keypoint_confidence = jnp.take(keypoint_confidence, sample_idx, axis=1)
        else:
            # NOTE: we repeat the last frame into the future to allow fitting on all frames while having
            # something to predict out. the alternative would be having to discard the tail of the trial,
            # which we can still do with this code present
            keypoints = jnp.stack(
                [jnp.take(keypoints, jnp.clip(sample_idx + i, None, trial_len - 1), axis=1) for i in range(self.future_timestamps + 1)], axis=2
            )
            keypoint_confidence = jnp.stack(
                [jnp.take(keypoint_confidence, jnp.clip(sample_idx + i, None, trial_len - 1), axis=1) for i in range(self.future_timestamps + 1)],
                axis=2,
            )

        if self.multisession:
            return (jnp.array(session_idx), jnp.array(trial_idx), timestamps), (keypoints, keypoint_confidence)
        if not self.multisession:
            # do this to maintain backward compatibility
            return (jnp.array(idx), timestamps), (keypoints, keypoint_confidence)

    def get_all_timestamps(self, idx):
        assert idx < len(self.timestamps), f"Index {idx} out of range {len(self.timestamps)}"
        return self.timestamps[idx][: self.trial_lengths[idx]]

    def get_all_keypoints(self, idx):
        assert idx < len(self.keypoints), f"Index {idx} out of range {len(self.keypoints)}"
        return self.keypoints[idx][:, : self.trial_lengths[idx]], self.keypoint_confidence[idx][:, : self.trial_lengths[idx]]

    @property
    def num_sessions(self) -> int:
        # return the number of stacked datasets, or 0 if this is an unstacked dataset
        return jnp.where(len(self.timestamps.shape) == 2, 1, self.timestamps.shape[0])

    @property
    def multisession(self) -> int:
        # return the number of stacked datasets, or 0 if this is an unstacked dataset
        # return jnp.where(len(self.timestamps.shape) == 2, jnp.array(False), jnp.array(True))
        return len(self.timestamps.shape) == 3  # not this is not returning a jax type to make sure using this is non-hashable

    # static method
    @staticmethod
    def stack_datasets(datasets: List["KeypointDataset"]) -> "KeypointDataset":

        def f(*args):
            return jnp.stack(list(args), axis=0)

        return jax.tree_util.tree_map(f, *datasets)

    def replace(self, **kwargs):
        """
        Replace fields in the dataset

        Uses the eqx.tree_at to edit the fields in the dataset.

        Parameters:
            **kwargs: The fields to replace

        Returns:
            KeypointDataset: The dataset with the fields replaced
        """

        for key, value in kwargs.items():
            where = lambda l: getattr(l, key)
            self = eqx.tree_at(where, self, value)

        return self

    def strip_keys(self):
        """When stacking multiple datasets we need to remove the static keys field"""

        # unfortunately we cannot implement this using .replace(keys=None) because that does not
        # work with static fields. instead we need to delve into the pytree and edit it directly

        # also this snippet doesn't work as the equinox module class makes this immutable
        import copy

        dataset_copy = copy.copy(self)
        object.__setattr__(dataset_copy, "keys", None)
        return dataset_copy


def level_floor_from_dataset(
    dataset: KeypointDataset, camera_params: Dict, z_offset: Float = 0.0, confidence_threshold: Float = 0.25, sigma: Float = 150.0
) -> KeypointDataset:
    """
    Level the floor from a dataset of keypoints.


    """
    from multi_camera.analysis.camera import robust_triangulate_points
    from multi_camera.analysis.calibration import shift_calibration
    from scipy.linalg import svd
    import numpy as np

    kp3d = []
    for i in range(len(dataset)):
        kp2d, kpc = dataset.get_all_keypoints(i)
        kp2d = jnp.concatenate([kp2d, kpc[..., None]], axis=-1)
        _kp3d = robust_triangulate_points(camera_params, kp2d, return_weights=True, threshold=confidence_threshold, sigma=sigma)[0]

        kp3d.append(_kp3d)

    # concatenate the 3D keypoints from all the recordings
    kp3d = np.concatenate(kp3d, axis=0)

    # filter out keypoints where all the joints are visible
    found = np.all(kp3d[:, :, 3] > 0.5, axis=-1)
    kp3d_found = kp3d[found, :, :3]

    # find the lowest point for each frame (presumes up is broadly correct)
    lowest = np.argmin(kp3d_found[:, :, 2], axis=1)
    lowest_kp3d = kp3d_found[np.arange(len(lowest)), lowest]

    def define_plane(points):
        """
        Defines a plane using an array of 3D points.

        Parameters:
            points (numpy.ndarray): An N x 3 matrix of coordinates.

        Returns:
            tuple:
                origin (numpy.ndarray): The centroid of the points.
                vector1 (numpy.ndarray): The first principal component.
                vector2 (numpy.ndarray): The second principal component.
        """
        # Compute the centroid
        origin = np.mean(points, axis=0)

        # Center the points
        points_centered = points - origin

        # Compute SVD
        U, _, _ = svd(points_centered.T, full_matrices=False)

        # Extract the first two principal components, then define coordinate system
        x = U[:, 0]
        y = U[:, 1]
        z = np.cross(x, y)

        floor_orientation = np.stack([x, y, z])

        return origin, floor_orientation

    # estimate the floor place and origin
    origin, floor_orientation = define_plane(lowest_kp3d)

    camera_params_leveled = shift_calibration(camera_params, origin, floor_orientation, z_offset)

    return camera_params_leveled

class MonocularDataset(KeypointDataset):

    """Largely simlar to KeypointDataset, but with a few additional features for monocular data including:
    (1) 3D keypoints and (2) phone attitude."""

    keypoints_3d: Float[Array, "trials 1 time joints 3"]
    phone_attitude: Float[Array, "trials 1 time 4"] | None
    attitude_lengths: Float[Array, "trials"] | None
    attitude_sample_length: int = 1
    keys: List
    timestamps_original: List
    camera_params: Dict

    def __init__(
        self,
        timestamps: Array,
        keypoints_2d: Array,
        keypoints_3d: Array,
        keypoint_confidence: Array,
        camera_params: Dict,
        phone_attitude: Array | None = None,
        sample_length=None,
        align_timestamps=True,
        max_length=None,
        future_timestamps=-1,
        keys=None,
    ):
        # initialize the parent class
        super().__init__(
            timestamps,
            keypoints_2d,
            keypoint_confidence,
            sample_length,
            align_timestamps,
            max_length,
            future_timestamps,
        )
        self.camera_params = camera_params
        self.keys = keys

        timestamps_original = copy.deepcopy(timestamps)

        if align_timestamps:
            for i in range(len(timestamps)):
                if phone_attitude is not None:
                    t0 = timestamps[i][np.abs(timestamps[i] - phone_attitude[i][0,0]).argmin()]

                    phone_attitude[i][:, 0] -= t0
                    timestamps[i] -= t0

                    # mask video based on phone attitude
                    ts_mask = np.logical_and.reduce((timestamps[i] >= 0, timestamps[i] <= phone_attitude[i][-1, 0]))
                    timestamps[i] = timestamps[i][ts_mask]
                    timestamps_original[i] = timestamps_original[i][ts_mask]
                    keypoints_2d[i] = keypoints_2d[i][:, ts_mask]
                    keypoints_3d[i] = keypoints_3d[i][:, ts_mask]
                    keypoint_confidence[i] = keypoint_confidence[i][:, ts_mask]

                    # mask phone additude based on video
                    attitude_mask = np.logical_and.reduce((phone_attitude[i][:, 0] >= 0, phone_attitude[i][:, 0] <= timestamps[i][-1]))
                    phone_attitude[i] = phone_attitude[i][attitude_mask]

                    assert phone_attitude[i][0, 0] >= 0, "first timestamp is not positive"
                else:
                    t0 = timestamps[i][0]
                    timestamps[i] -= t0
                assert timestamps[i][0] >= 0, "first timestamp is not positive"

        for ta, to in zip(timestamps, timestamps_original):
            assert len(ta) == len(to), "timestamps and original timestamps are not the same length"

        self.timestamps_original = timestamps_original

        if max_length is None:
            max_length = max([len(t) for t in timestamps])

        if phone_attitude is not None and sample_length is None:
            max_attitude_length = max([g.shape[0] for g in phone_attitude])
            self.attitude_sample_length = max_attitude_length
            self.attitude_lengths = jnp.array([g.shape[0] for g in phone_attitude])
        elif phone_attitude is not None:
            max_attitude_length = max([g.shape[0] for g in phone_attitude])
            self.attitude_lengths = jnp.array([g.shape[0] for g in phone_attitude])
            self.attitude_sample_length = sample_length

        # this is defined in the init of KeypointDataset
        def padded_stack(y, time_axis=0, max_length=max_length):

            def trim(_y):
                if _y.shape[time_axis] > max_length:
                    _y = jnp.take(_y, jnp.arange(max_length), axis=time_axis)
                return _y

            y = [trim(_y) for _y in y]

            shape = list(y[0].shape)
            shape[time_axis] = max_length
            x = jnp.zeros((len(y), *shape))
            for i, t in enumerate(y):
                if time_axis == 0:
                    x = x.at[i, : t.shape[0]].set(t)
                elif time_axis == 1:
                    x = x.at[i, :, : t.shape[1]].set(t)
                else:
                    raise NotImplementedError("Only time axis 0 or 1 supported")
            return x

        # add monocular-specific data
        self.keypoints = padded_stack(keypoints_2d, 1)
        self.keypoints_3d = padded_stack(keypoints_3d, 1)
        self.keypoint_confidence = padded_stack(keypoint_confidence, 1)
        self.timestamps = padded_stack(timestamps)
        if phone_attitude is not None:
            self.phone_attitude = padded_stack(phone_attitude, 0, max_attitude_length)  # unsure about which axis here
        else:
            self.phone_attitude = None
            self.attitude_lengths = None
        self.trial_lengths = jnp.array([min(len(t), max_length) for t in timestamps])

    def _internal_getitem(self, idx, sample_length):
        # idx gets modulo'd to index into multiple flattened dimensions. first it indexes into trials within a session
        # then it wraps around and indexes sessions, and finally it wraps around and residual determines the start
        # offset

        total_trials = self.num_trials * self.num_sessions
        start_idx = idx // total_trials  # offset for start
        idx = (idx % total_trials).astype(int)
        session_idx = idx // self.num_trials
        trial_idx = idx % self.num_trials

        # handle where we have multiple sessions or a single session in a consistent manner
        if self.multisession:
            session_fetch = lambda x: jnp.take(x, session_idx, axis=0)
        else:
            session_fetch = lambda x: x

        session_trial_len = session_fetch(self.trial_lengths)
        session_timestamps = session_fetch(self.timestamps)
        session_keypoints_2d = session_fetch(self.keypoints)
        session_keypoints_3d = session_fetch(self.keypoints_3d)
        if self.phone_attitude is not None:
            session_attitude = session_fetch(self.phone_attitude)

        session_keypoint_confidence = session_fetch(self.keypoint_confidence)

        trial_len = jnp.take(session_trial_len, trial_idx)
        timestamps = jnp.take(session_timestamps, trial_idx, axis=0)
        keypoints_2d = jnp.take(session_keypoints_2d, trial_idx, axis=0)
        keypoints_3d = jnp.take(session_keypoints_3d, trial_idx, axis=0)
        keypoint_confidence = jnp.take(session_keypoint_confidence, trial_idx, axis=0)

        if self.phone_attitude is not None:
            phone_attitude = jnp.take(session_attitude, trial_idx, axis=0)
            attitude_len = jnp.take(self.attitude_lengths, trial_idx)

        # compute deterministically interleaved samples
        N = jnp.ceil(trial_len / sample_length).astype(int)

        sample_idx = jnp.arange(sample_length) * N + start_idx
        sample_idx = (sample_idx % trial_len).astype(int)

        # compute samples for attitudescope
        if self.phone_attitude is not None:
            N = jnp.ceil(attitude_len / self.attitude_sample_length).astype(int)
            attitude_sample_idx = jnp.arange(self.attitude_sample_length) * N + start_idx
            attitude_sample_idx = (attitude_sample_idx % attitude_len).astype(int)

        timestamps = jnp.take(timestamps, sample_idx, axis=0)
        if self.future_timestamps == -1:
            keypoints_2d = jnp.take(keypoints_2d, sample_idx, axis=1)
            keypoints_3d = jnp.take(keypoints_3d, sample_idx, axis=1)
            keypoint_confidence = jnp.take(keypoint_confidence, sample_idx, axis=1)

            if self.phone_attitude is not None:
                phone_attitude = jnp.take(phone_attitude, attitude_sample_idx, axis=0)
                attitude_timestamps = phone_attitude[:, 0]
                attitude_data = phone_attitude[:, 1:]
        else:
            raise NotImplementedError("Future timestamps not implemented for monocular")
            keypoints_2d = jnp.stack([jnp.take(keypoints, sample_idx + i, axis=1) for i in range(self.future_timestamps + 1)], axis=2)
            keypoint_confidence = jnp.stack(
                [jnp.take(keypoint_confidence, sample_idx + i, axis=1) for i in range(self.future_timestamps + 1)], axis=2
            )

        if self.multisession:
            return (jnp.array(session_idx), jnp.array(trial_idx), timestamps), (keypoints_2d, keypoint_confidence)
        if not self.multisession:
            # do this to maintain backward compatibility
            if self.phone_attitude is not None:
                return (jnp.array(idx), timestamps, attitude_timestamps), (keypoints_2d, keypoints_3d, keypoint_confidence, attitude_data)
            else:
                return (jnp.array(idx), timestamps, None), (
                    keypoints_2d,
                    keypoints_3d,
                    keypoint_confidence,
                    None,
                )  # just empty attitude for compatibility

    def get_all_keypoints_3d(self, idx):
        assert idx < self.keypoints_3d.shape[0], "Index out of bounds"
        return self.keypoints_3d[idx][:, : self.trial_lengths[idx]].squeeze(), self.keypoint_confidence[idx][:, : self.trial_lengths[idx]].squeeze()

    def get_all_attitude(self, idx):
        assert idx < self.phone_attitude.shape[0], "Index out of bounds"
        return (
            self.phone_attitude[idx][: self.attitude_lengths[idx]][:, 0].squeeze(),
            self.phone_attitude[idx][: self.attitude_lengths[idx]][:, 1:].squeeze(),
        )

    def get_all_original_timestamps(self, idx):
        assert idx < len(self.timestamps_original), "Index out of bounds"
        return self.timestamps_original[idx]

def get_samsung_calibration() -> Dict:
    """Gives the calibration params in the MultiCameraTracking format for our S20. I got these from a checkerboard video I filmed. Not sure how different this is between S20 cameras. Includes no distortion."""
    mtx = np.array(
        [[1.43333476e03, 0.00000000e00, 5.39500000e02], [0.00000000e00, 1.43333476e03, 9.59500000e02], [0.00000000e00, 0.00000000e00, 1.00000000e00]]
    )
    mtx = (mtx[[0, 1, 0, 1], [0, 1, 2, 2]] / 1000).reshape(1, -1)
    dist = np.zeros(5).reshape(1, -1)
    rvec = np.zeros(3).reshape(1, -1)
    tvec = np.zeros(3).reshape(1, -1)
    return dict(mtx=mtx, dist=dist, tvec=tvec, rvec=rvec)
