"""
Compute MoVi error metrics from pre-aligned Euler qpos files in data/movi/.

Both marker_qpos_euler and video_qpos_aligned are (T, 40) Euler format.

  MJAE  — body joints only (cols 6+), bilateral joints grouped.
  MAE   — SO(3)-aligned pelvis orientation + lower body, wrap_to_pi on differences.
  RTE   — Euclidean distance of translation cols after Procrustes (already baked in).

Usage: python analysis/compute_movi_errors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[1] / "data" / "movi"

# ---------------------------------------------------------------------------
# Joint selection
# qpos column order for humanoid_torque.xml (40-col Euler):
# ['pelvis_tx','pelvis_tz','pelvis_ty','pelvis_tilt','pelvis_list','pelvis_rotation',
#  'hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r',
#  'subtalar_angle_r','mtp_angle_r','hip_flexion_l','hip_adduction_l','hip_rotation_l',
#  'knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l','lumbar_extension',
#  'lumbar_bending','lumbar_rotation','neck_extension','neck_bending','neck_rotation',
#  'arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r','pro_sup_r','wrist_flex_r',
#  'wrist_dev_r','arm_flex_l','arm_add_l','arm_rot_l','elbow_flex_l','pro_sup_l',
#  'wrist_flex_l','wrist_dev_l']
# ---------------------------------------------------------------------------
MJAE_JOINTS = [
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "neck_extension", "neck_bending", "neck_rotation",
    "arm_flex_r", "arm_add_r", "elbow_flex_r",
    "arm_flex_l", "arm_add_l", "elbow_flex_l",
]

MAE_JOINTS = [
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
    "knee_angle_r", "ankle_angle_r", "subtalar_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
    "knee_angle_l", "ankle_angle_l", "subtalar_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
]

_JOINT_NAMES = [
    "pelvis_tx", "pelvis_tz", "pelvis_ty",
    "pelvis_tilt", "pelvis_list", "pelvis_rotation",
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
    "subtalar_angle_r", "mtp_angle_r",
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    "subtalar_angle_l", "mtp_angle_l",
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "neck_extension", "neck_bending", "neck_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l",
]

MJAE_KEEP_INDS = np.array([_JOINT_NAMES.index(j) for j in MJAE_JOINTS])
MAE_KEEP_INDS = np.array([_JOINT_NAMES.index(j) for j in MAE_JOINTS])

GROUPED_JOINT_INDS, GROUPED_JOINT_NAMES = [], []
for joint in MJAE_JOINTS:
    if joint.endswith("_r"):
        GROUPED_JOINT_INDS.append([MJAE_JOINTS.index(joint), MJAE_JOINTS.index(joint[:-2] + "_l")])
        GROUPED_JOINT_NAMES.append(joint[:-2])
    elif joint.endswith("_l"):
        continue
    else:
        GROUPED_JOINT_INDS.append(MJAE_JOINTS.index(joint))
        GROUPED_JOINT_NAMES.append(joint)
GROUPED_JOINT_NAMES.append("All")
GROUPED_JOINT_INDS.append(list(range(len(MJAE_JOINTS))))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def per_trial_mjae(marker: np.ndarray, video: np.ndarray) -> np.ndarray:
    """Per-trial median absolute error grouped into bilateral joints (deg)."""
    diff = np.abs(wrap_to_pi(video[:, MJAE_KEEP_INDS] - marker[:, MJAE_KEEP_INDS]))
    errors = []
    for inds in GROUPED_JOINT_INDS:
        errors.append(np.median(diff[:, np.array(inds)]))
    return np.array(errors) * 180 / np.pi


def per_trial_mae(marker: np.ndarray, video: np.ndarray) -> np.ndarray:
    """Per-trial mean absolute error per DOF (deg)."""
    diff = np.abs(wrap_to_pi(marker[:, MAE_KEEP_INDS] - video[:, MAE_KEEP_INDS]))
    return diff.mean(axis=0) * 180 / np.pi


def per_trial_rte_cm(marker: np.ndarray, video: np.ndarray) -> float:
    """Median root translation error in cm from aligned translation cols."""
    dist = np.linalg.norm(video[:, :3] - marker[:, :3], axis=-1)
    return np.median(dist) * 100


def format_joint_name(name: str) -> str:
    if name == "knee_angle":
        return "Knee Flexion"
    return name.replace("_", " ").title()


def mjae_df(per_trial: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(GROUPED_JOINT_NAMES):
        med = np.median(per_trial[:, i])
        niqr = (np.percentile(per_trial[:, i], 75) - np.percentile(per_trial[:, i], 25)) * 0.7413
        rows.append({"Joint": name, "Error (nIQR)": f"{med:.2f} ({niqr:.2f})"})
    df = pd.DataFrame(rows)
    df["Joint"] = df["Joint"].apply(format_joint_name)
    return df.set_index("Joint")


# ---------------------------------------------------------------------------
# Load aligned files and compute
# ---------------------------------------------------------------------------
files = sorted(DATA_DIR.glob("participant_*.npz"))
if not files:
    raise FileNotFoundError(f"No participant .npz files found in {DATA_DIR}")

mjae_trials, mae_trials, rte_trials = [], [], []
for path in files:
    data = np.load(path)
    marker = data["marker_qpos_euler"]   # (T, 40)
    video = data["video_qpos_aligned"]   # (T, 40)
    mjae_trials.append(per_trial_mjae(marker, video))
    mae_trials.append(per_trial_mae(marker, video))
    rte_trials.append(per_trial_rte_cm(marker, video))

mjae_arr = np.array(mjae_trials)
mae_arr = np.vstack(mae_trials)
rte_arr = np.array(rte_trials)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print("=" * 50)
print("MJAE — median (nIQR) in degrees")
print("=" * 50)
print(mjae_df(mjae_arr).to_string())

mae_mean = mae_arr.mean(axis=0)
mae_std = mae_arr.std(axis=0)
mae_table = pd.DataFrame({"Mean (deg)": mae_mean, "Std (deg)": mae_std}, index=MAE_JOINTS)
print()
print("=" * 50)
print("MAE (SO(3)-aligned root) — mean ± std in degrees")
print("=" * 50)
print(mae_table.to_string())
print(f"\nGrand mean: {mae_mean.mean():.2f} ± {mae_mean.std():.2f} deg")

rte_median = np.median(rte_arr)
rte_niqr = (np.percentile(rte_arr, 75) - np.percentile(rte_arr, 25)) * 0.74
print()
print("=" * 50)
print("RTE — root translation error")
print("=" * 50)
print(f"Median (nIQR): {rte_median:.2f} ({rte_niqr:.2f}) cm")
