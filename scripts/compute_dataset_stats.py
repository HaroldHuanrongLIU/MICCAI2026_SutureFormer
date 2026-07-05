"""Compute and save dataset statistics for normalization."""

import json
import os
from collections import defaultdict

import numpy as np

DATA_ROOT = (
    "/path/to/dataset/"
    "MICCAI2026_Surgical_Trajectory_Prediction_Dataset"
)
OUTPUT_PATH = "data/stats/dataset_statistics.json"


def main():
    all_x, all_y = [], []
    all_speeds = []
    traj_lengths = []
    patient_traj_counts = defaultdict(int)

    patients = sorted(
        d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    print(f"Found {len(patients)} patients")

    for patient_id in patients:
        patient_dir = os.path.join(DATA_ROOT, patient_id)
        for traj_id in sorted(os.listdir(patient_dir)):
            traj_dir = os.path.join(patient_dir, traj_id)
            if not os.path.isdir(traj_dir):
                continue

            label_file = os.path.join(traj_dir, f"{patient_id}_{traj_id}_labels.json")
            if not os.path.exists(label_file):
                continue

            with open(label_file) as f:
                data = json.load(f)

            labels = data["labels"]
            patient_traj_counts[patient_id] += 1
            traj_lengths.append(len(labels))

            for i, lbl in enumerate(labels):
                all_x.append(lbl["pos_x"])
                all_y.append(lbl["pos_y"])

                if i > 0:
                    dx = lbl["pos_x"] - labels[i - 1]["pos_x"]
                    dy = lbl["pos_y"] - labels[i - 1]["pos_y"]
                    speed = np.sqrt(dx**2 + dy**2)
                    all_speeds.append(speed)

    all_x = np.array(all_x, dtype=np.float64)
    all_y = np.array(all_y, dtype=np.float64)
    all_speeds = np.array(all_speeds, dtype=np.float64)
    traj_lengths = np.array(traj_lengths)

    stats = {
        "num_patients": len(patients),
        "num_trajectories": sum(patient_traj_counts.values()),
        "total_frames": len(all_x),
        "coordinate_stats": {
            "x_mean": float(all_x.mean()),
            "x_std": float(all_x.std()),
            "x_min": float(all_x.min()),
            "x_max": float(all_x.max()),
            "y_mean": float(all_y.mean()),
            "y_std": float(all_y.std()),
            "y_min": float(all_y.min()),
            "y_max": float(all_y.max()),
        },
        "speed_stats": {
            "mean": float(all_speeds.mean()),
            "std": float(all_speeds.std()),
            "median": float(np.median(all_speeds)),
            "p95": float(np.percentile(all_speeds, 95)),
            "max": float(all_speeds.max()),
        },
        "trajectory_length_stats": {
            "mean": float(traj_lengths.mean()),
            "std": float(traj_lengths.std()),
            "median": float(np.median(traj_lengths)),
            "min": int(traj_lengths.min()),
            "max": int(traj_lengths.max()),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to {OUTPUT_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
