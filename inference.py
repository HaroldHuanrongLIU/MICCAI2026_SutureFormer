"""Single-trajectory inference and visualization script."""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from sutureformer.datasets.crop import extract_local_crop
from sutureformer.datasets.guidance import construct_guidance_channel
from sutureformer.datasets.keyframe_split import compute_obs_pred_split
from sutureformer.models.sutureformer import SutureFormer
from sutureformer.utils.direction_map import DIRECTION_VECTORS
from sutureformer.utils.guidance_extrapolation import generate_test_guidance

# Coordinate normalization constants
X_MAX = 1263.0
Y_MAX = 901.0


def main():
    parser = argparse.ArgumentParser(description="SutureFormer single-trajectory inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/path/to/dataset/"
        "MICCAI2026_Surgical_Trajectory_Prediction_Dataset",
    )
    parser.add_argument("--patient", type=str, required=True, help="Patient ID (e.g., '021')")
    parser.add_argument("--trajectory", type=str, required=True, help="Trajectory ID (e.g., '402')")
    parser.add_argument(
        "--obs_keyframe_end", type=int, default=6, help="Number of keyframes for obs phase"
    )
    parser.add_argument("--output", type=str, default="outputs/inference_result.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SutureFormer()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load trajectory
    traj_dir = os.path.join(args.data_root, args.patient, args.trajectory)
    label_file = os.path.join(traj_dir, f"{args.patient}_{args.trajectory}_labels.json")
    with open(label_file) as f:
        label_data = json.load(f)
    labels = label_data["labels"]

    # Compute keyframe-based split
    n_obs, n_pred = compute_obs_pred_split(labels, args.obs_keyframe_end)
    if n_obs == 0:
        print("Error: trajectory has insufficient keyframes for split")
        return

    print(f"Trajectory: {len(labels)} frames, n_obs={n_obs}, n_pred={n_pred}")

    # Only use observation-phase coordinates for guidance (no future leakage)
    obs_coords_px = [(lbl["pos_x"], lbl["pos_y"]) for lbl in labels[:n_obs]]
    obs_confidences = [lbl["confidence"] for lbl in labels[:n_obs]]

    # Build observation tensors (pre-concatenated 4-channel)
    obs_images, obs_coords = [], []
    for lbl in labels[:n_obs]:
        x, y = lbl["pos_x"], lbl["pos_y"]
        img = np.array(Image.open(os.path.join(traj_dir, lbl["frame_file"])).convert("RGB"))
        crop = extract_local_crop(img, x, y, 128)
        crop_tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
        guid = construct_guidance_channel(x, y, obs_coords_px, obs_confidences, 128)
        guid_tensor = torch.from_numpy(guid).float().permute(2, 0, 1)
        combined = torch.cat([crop_tensor, guid_tensor], dim=0)  # [4, 128, 128]
        obs_images.append(combined)
        obs_coords.append(torch.tensor([x / X_MAX, y / Y_MAX], dtype=torch.float32))

    obs_images_t = torch.stack(obs_images).unsqueeze(0).to(device)  # [1, n_obs, 4, 128, 128]
    obs_coords_t = torch.stack(obs_coords).unsqueeze(0).to(device)  # [1, n_obs, 2]
    obs_mask = torch.ones(1, n_obs, dtype=torch.bool, device=device)  # [1, n_obs]
    n_pred_t = torch.tensor([n_pred], device=device)

    # Inference
    with torch.no_grad():
        z_ctx = model.encode_observation(obs_images_t, obs_coords_t, obs_mask)
        guidance = generate_test_guidance(obs_coords_t, obs_mask, n_pred_t)

        current_pos = obs_coords_t[:, -1, :]
        predicted = []

        dirs = DIRECTION_VECTORS.to(device)  # [9, 2]
        for k in range(n_pred):
            step_ratio = torch.tensor([[k / n_pred]], device=device)
            logits, mag = model.predict_step(z_ctx, current_pos, guidance[:, k, :], step_ratio)
            probs = torch.softmax(logits, dim=-1)  # [1, 9]
            weighted_dir = probs @ dirs  # [1, 2]
            displacement = weighted_dir * mag  # [1, 2]
            next_pos = (current_pos + displacement).clamp(0.0, 1.0)
            predicted.append(next_pos)
            current_pos = next_pos

    predicted = torch.cat(predicted, dim=0).cpu().numpy()  # [n_pred, 2]
    pred_px = predicted.copy()
    pred_px[:, 0] *= X_MAX
    pred_px[:, 1] *= Y_MAX

    # Ground truth prediction coordinates
    gt_coords = np.array([[lbl["pos_x"], lbl["pos_y"]] for lbl in labels[n_obs : n_obs + n_pred]])
    obs_px = np.array([[lbl["pos_x"], lbl["pos_y"]] for lbl in labels[:n_obs]])

    # Visualize on last observation image
    last_img = np.array(
        Image.open(os.path.join(traj_dir, labels[n_obs - 1]["frame_file"])).convert("RGB")
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(last_img)
    if len(obs_px) > 0:
        ax.plot(obs_px[:, 0], obs_px[:, 1], "w-", linewidth=1.5, alpha=0.6)
        ax.scatter(obs_px[:, 0], obs_px[:, 1], c="white", s=30, zorder=4,
                   edgecolors="gray", linewidths=0.5)
    ax.plot(gt_coords[:, 0], gt_coords[:, 1], "g-", linewidth=2.5, label="GT")
    ax.scatter(gt_coords[:, 0], gt_coords[:, 1], c="lime", s=50, zorder=5,
               edgecolors="darkgreen", linewidths=0.8)
    ax.plot(pred_px[:, 0], pred_px[:, 1], "r-", linewidth=2.5, label="Predicted")
    ax.scatter(pred_px[:, 0], pred_px[:, 1], c="red", s=50, zorder=5,
               edgecolors="darkred", linewidths=0.8)
    ax.set_title(f"Patient {args.patient}, Trajectory {args.trajectory}", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to {args.output}")

    # Print metrics
    from sutureformer.metrics.ade import compute_ade
    from sutureformer.metrics.fde import compute_fde

    pred_t = torch.from_numpy(pred_px).unsqueeze(0)
    gt_t = torch.from_numpy(gt_coords.astype(np.float32)).unsqueeze(0)
    mask = torch.ones(1, n_pred, dtype=torch.bool)
    n_pred_cpu = torch.tensor([n_pred])
    print(f"ADE: {compute_ade(pred_t, gt_t, mask):.2f} px")
    print(f"FDE: {compute_fde(pred_t, gt_t, n_pred_cpu):.2f} px")


if __name__ == "__main__":
    main()
