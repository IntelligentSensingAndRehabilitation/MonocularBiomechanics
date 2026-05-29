"""
Reproduce the monocular and static error tables from monocular_paper_figures_rev_export.ipynb
without any DataJoint or body_models dependencies.

The per-trial errors are already computed and stored in data/clinical/:
  monocular_combined_joint_errors.csv
  static_combined_joint_errors.csv

  *_median columns  — per-trial MJAE (bilateral-grouped median absolute error, degrees)
  *_mean columns    — per-trial MAE per DOF (mean absolute error, degrees)
  RTE (cm)          — root translation error in cm

Usage: python analysis/compute_clinical_errors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[1] / "data" / "clinical"

# Joint order matches GROUPED_JOINT_NAMES from the notebook
MJAE_COLS = [
    "hip_flexion_median",
    "hip_adduction_median",
    "hip_rotation_median",
    "knee_angle_median",
    "ankle_angle_median",
    "lumbar_extension_median",
    "lumbar_bending_median",
    "lumbar_rotation_median",
    "neck_extension_median",
    "neck_bending_median",
    "neck_rotation_median",
    "arm_flex_median",
    "arm_add_median",
    "elbow_flex_median",
    "All_median",
]

MAE_COLS = [
    "pelvis_tilt_mean",
    "pelvis_list_mean",
    "pelvis_rotation_mean",
    "hip_flexion_r_mean",
    "hip_adduction_r_mean",
    "hip_rotation_r_mean",
    "knee_angle_r_mean",
    "ankle_angle_r_mean",
    "subtalar_angle_r_mean",
    "hip_flexion_l_mean",
    "hip_adduction_l_mean",
    "hip_rotation_l_mean",
    "knee_angle_l_mean",
    "ankle_angle_l_mean",
    "subtalar_angle_l_mean",
    "lumbar_extension_mean",
    "lumbar_bending_mean",
    "lumbar_rotation_mean",
]


def format_joint_name(name: str) -> str:
    if name == "knee_angle":
        return "Knee Flexion"
    return name.replace("_", " ").title()


def mjae_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MJAE_COLS:
        vals = df[col].dropna().values
        joint = col.replace("_median", "")
        med = np.median(vals)
        niqr = (np.percentile(vals, 75) - np.percentile(vals, 25)) * 0.7413
        rows.append({"Joint": format_joint_name(joint), "Error (nIQR)": f"{med:.2f} ({niqr:.2f})"})
    return pd.DataFrame(rows).set_index("Joint")


def mae_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in MAE_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        joint = col.replace("_mean", "")
        mean = np.mean(vals)
        std = np.std(vals)
        rows.append({"Joint": joint, "Mean (deg)": f"{mean:.2f}", "Std (deg)": f"{std:.2f}"})
    return pd.DataFrame(rows).set_index("Joint")


def rte_summary(df: pd.DataFrame) -> str:
    if "RTE (cm)" not in df.columns:
        return "RTE: N/A"
    vals = df["RTE (cm)"].dropna().values
    med = np.median(vals)
    niqr = (np.percentile(vals, 75) - np.percentile(vals, 25)) * 0.74
    return f"Median (nIQR): {med:.2f} ({niqr:.2f}) cm"


def print_table(label: str, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Trials: {len(df)}  |  Subjects: {df['subject_hash'].nunique()}")

    print("\nMJAE — median (nIQR) in degrees")
    print("-" * 40)
    print(mjae_summary(df).to_string())

    print("\nMAE — mean ± std in degrees")
    print("-" * 40)
    mae = mae_summary(df)
    print(mae.to_string())
    all_means = df[[c for c in MAE_COLS if c in df.columns]].mean(axis=1)
    grand_mean = all_means.mean()
    grand_std = all_means.std()
    print(f"\nGrand mean: {grand_mean:.2f} ± {grand_std:.2f} deg")

    print("\nRTE — root translation error")
    print("-" * 40)
    print(rte_summary(df))


print_table("MONOCULAR (dynamic)", DATA_DIR / "monocular_combined_joint_errors.csv")
print_table("STATIC", DATA_DIR / "static_combined_joint_errors.csv")
