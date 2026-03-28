import csv
import json
import logging
import os
import pdb
import time
from os import path as osp

import numpy as np
import torch
import wandb
from evaluation import postprocess
from evaluation.metrics import compute_accruacy_metrics, compute_ate_rte
from model import function
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import logging_config
from utils.rich_logging import error, info, rich_logger, success, warning

# from dataloader import dataset as dataset_utils
console = logging_config.Console()


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def segment_trajectory_5m(traj_attr_dict, segment_length=5.0):
    """
    Segment trajectory into 5-meter chunks for local accuracy evaluation.

    Args:
        traj_attr_dict: Dictionary containing trajectory data
        segment_length: Length of each segment in meters (default: 5.0)

    Returns:
        List of trajectory segments, each containing data for ~5m of travel
    """
    segments = []

    # Get ground truth positions
    pos_gt = traj_attr_dict["pos_gt"]
    ts = traj_attr_dict["ts"]

    # Calculate cumulative distance from start
    cumulative_distance = np.zeros(len(pos_gt))
    for i in range(1, len(pos_gt)):
        step_distance = np.linalg.norm(pos_gt[i] - pos_gt[i - 1])
        cumulative_distance[i] = cumulative_distance[i - 1] + step_distance

    # Find segment boundaries
    segment_boundaries = []
    current_distance = 0.0

    for i in range(len(cumulative_distance)):
        if cumulative_distance[i] >= current_distance + segment_length:
            segment_boundaries.append(i)
            current_distance = cumulative_distance[i]

    # Add final boundary if not already included
    if len(segment_boundaries) == 0 or segment_boundaries[-1] != len(pos_gt) - 1:
        segment_boundaries.append(len(pos_gt) - 1)

    # Create segments
    start_idx = 0
    for end_idx in segment_boundaries:
        if end_idx > start_idx:  # Ensure segment has at least 2 points
            segment = {}
            for key in traj_attr_dict.keys():
                if isinstance(traj_attr_dict[key], np.ndarray):
                    segment[key] = traj_attr_dict[key][start_idx : end_idx + 1]
                else:
                    segment[key] = traj_attr_dict[key]

            # Calculate actual segment length
            segment_actual_length = (
                cumulative_distance[end_idx] - cumulative_distance[start_idx]
            )
            segment["segment_length"] = segment_actual_length
            segment["start_idx"] = start_idx
            segment["end_idx"] = end_idx

            segments.append(segment)
            start_idx = end_idx

    return segments


def segment_trajectory_by_distance_gt(
    gt_positions, gt_timestamps, imu_timestamps, segment_length=5.0
):
    """
    Segment trajectory into 5-meter chunks based on ground truth odometry.
    Find corresponding IMU data indices for each segment.

    Args:
        gt_positions: Ground truth positions (N, 3)
        gt_timestamps: Ground truth timestamps (N,)
        imu_timestamps: IMU timestamps (M,)
        segment_length: Length of each segment in meters (default: 5.0)

    Returns:
        List of segment dictionaries containing:
        - gt_start_idx, gt_end_idx: Ground truth indices
        - imu_start_idx, imu_end_idx: IMU indices
        - initial_pose: First pose of the segment
        - segment_length: Actual segment length
    """
    segments = []

    # Calculate cumulative distance from start
    cumulative_distance = np.zeros(len(gt_positions))
    for i in range(1, len(gt_positions)):
        step_distance = np.linalg.norm(gt_positions[i] - gt_positions[i - 1])
        cumulative_distance[i] = cumulative_distance[i - 1] + step_distance

    # Find segment boundaries
    segment_boundaries = []
    current_distance = 0.0

    for i in range(len(cumulative_distance)):
        if cumulative_distance[i] >= current_distance + segment_length:
            segment_boundaries.append(i)
            current_distance = cumulative_distance[i]

    # Add final boundary if not already included
    if len(segment_boundaries) == 0 or segment_boundaries[-1] != len(gt_positions) - 1:
        segment_boundaries.append(len(gt_positions) - 1)

    # Create segments
    start_idx = 0
    for end_idx in segment_boundaries:
        if end_idx > start_idx:  # Ensure segment has at least 2 points
            # Get ground truth timestamps for this segment
            segment_gt_start_time = gt_timestamps[start_idx]
            segment_gt_end_time = gt_timestamps[end_idx]

            # Find corresponding IMU indices using timestamp overlap
            imu_start_idx = np.searchsorted(
                imu_timestamps, segment_gt_start_time, side="left"
            )
            imu_end_idx = np.searchsorted(
                imu_timestamps, segment_gt_end_time, side="right"
            )

            # Ensure we have valid IMU data
            if imu_start_idx < len(imu_timestamps) and imu_end_idx > imu_start_idx:
                # Calculate actual segment length
                segment_actual_length = (
                    cumulative_distance[end_idx] - cumulative_distance[start_idx]
                )

                # Get initial pose (first pose of the segment)
                initial_pose = gt_positions[start_idx]

                segment = {
                    "gt_start_idx": start_idx,
                    "gt_end_idx": end_idx,
                    "imu_start_idx": imu_start_idx,
                    "imu_end_idx": imu_end_idx,
                    "gt_start_time": segment_gt_start_time,
                    "gt_end_time": segment_gt_end_time,
                    "initial_pose": initial_pose,
                    "segment_length": segment_actual_length,
                    "cumulative_distance_start": cumulative_distance[start_idx],
                    "cumulative_distance_end": cumulative_distance[end_idx],
                }

                segments.append(segment)

            start_idx = end_idx

    return segments


def load_dataset_module(dataset_name):
    """
    Load the appropriate dataset module based on dataset name.

    Args:
        dataset_name: Name of the dataset ('RNIN', 'IDOL', 'TarTanAir', etc.)

    Returns:
        Dataset module
    """
    if dataset_name == "RNIN":
        from dataloader import dataset as dataset_utils

        info("Current Dataset is RNIN", style="cyan")
    elif dataset_name == "IDOL":
        from dataloader import dataset_IDOL as dataset_utils

        info("Current Dataset is IDOL", style="cyan")
    elif dataset_name == "TarTanAir":
        from dataloader import dataset_TartanAir as dataset_utils

        info("Current Dataset is TarTanAir", style="cyan")
    elif dataset_name == "TRO":
        from dataloader import dataset_TRO as dataset_utils

        info("Current Dataset is TRO dataset", style="cyan")
    elif dataset_name == "AirLab":
        from dataloader import dataset_AirLab as dataset_utils

        info("Current Dataset is AirLab", style="cyan")
    else:  # 'Euroc'
        from dataloader import dataset_Euroc as dataset_utils

    return dataset_utils


def create_test_dataloader(cfg, data_path):
    """
    Create test dataloader for a given data path.

    Args:
        cfg: Configuration dictionary
        data_path: Path to test data

    Returns:
        test_loader: DataLoader for testing
        test_dataset: Dataset object
    """
    dataset_utils = load_dataset_module(cfg["data"]["dataset"])

    test_basic_data = dataset_utils.BasicSequenceData(cfg, [data_path], mode="test")
    test_dataset = dataset_utils.SeqToSeqDataset(
        cfg, test_basic_data, test_basic_data.get_merged_index_map(), mode="test"
    )

    # Use train batch_size as fallback if test batch_size is not configured
    if "test" not in cfg or "batch_size" not in cfg["test"]:
        test_batch_size = cfg["train"]["batch_size"]
    else:
        test_batch_size = cfg["test"]["batch_size"]

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return test_loader, test_dataset


def create_output_directories(out_dir, subfolder_name, data_name):
    """
    Create output directories for test results.

    Args:
        out_dir: Base output directory
        subfolder_name: Subfolder name
        data_name: Data name

    Returns:
        outdir: Path to the output directory
    """
    sub_outdir = osp.join(out_dir, subfolder_name)
    if not osp.exists(sub_outdir):
        os.mkdir(sub_outdir)

    outdir = osp.join(sub_outdir, data_name)
    if not osp.exists(outdir):
        os.mkdir(outdir)

    return outdir


def compute_segment_metrics(segment, cfg):
    """
    Compute metrics for a single trajectory segment.

    Args:
        segment: Trajectory segment dictionary
        cfg: Configuration dictionary

    Returns:
        Dictionary containing all computed metrics
    """
    ate, t_rte, d_rte = compute_ate_rte(
        segment["pos_pred"], segment["pos_gt"], int(cfg["data"]["imu_freq"] * 1)
    )

    ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE = (
        compute_accruacy_metrics(
            segment, int(cfg["data"]["imu_freq"] * 1), cfg["data"]["use_local_coord"]
        )
    )

    return {
        "ate": ate,
        "t_rte": t_rte,
        "d_rte": d_rte,
        "ATE": ATE,
        "AVE": AVE,
        "P_RMSE": P_RMSE,
        "V_RMSE": V_RMSE,
        "X_ATE": X_ATE,
        "Y_ATE": Y_ATE,
        "Z_ATE": Z_ATE,
        "X_AVE": X_AVE,
        "Y_AVE": Y_AVE,
        "Z_AVE": Z_AVE,
    }


def process_trajectory_segments(trajectory_segments, outdir, cfg, trajectory_info=None):
    """
    Process all trajectory segments and compute metrics.

    Args:
        trajectory_segments: List of trajectory segments
        outdir: Output directory
        cfg: Configuration dictionary
        trajectory_info: Dictionary with trajectory metadata (data_name, robot_type, trajectory_id)

    Returns:
        List of segment metrics
    """
    segment_metrics = []

    for seg_idx, segment in enumerate(trajectory_segments):
        logging.info(
            f"Processing segment {seg_idx + 1}/{len(trajectory_segments)} (length: {segment['segment_length']:.2f}m)"
        )

        # Compute metrics for this segment
        metrics = compute_segment_metrics(segment, cfg)

        # Add segment metadata
        segment_metric = {
            "segment_id": seg_idx,
            "segment_length": segment["segment_length"],
            "start_idx": segment["start_idx"],
            "end_idx": segment["end_idx"],
            **metrics,
        }

        # Add trajectory information if provided
        if trajectory_info:
            segment_metric.update(trajectory_info)

        segment_metrics.append(segment_metric)

        # Save segment data
        save_segment_data(segment, segment_metric, outdir, seg_idx)

    return segment_metrics


def sanity_check_segment_alignment(segment, seg_idx):
    """
    Sanity check to ensure segment inference starts from ground truth initial position.

    Args:
        segment: Trajectory segment data
        seg_idx: Segment index

    Returns:
        bool: True if alignment is correct, False otherwise
    """
    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]

    # Check initial position alignment
    gt_initial = pos_gt[0]
    pred_initial = pos_pred[0]
    initial_diff = np.linalg.norm(pred_initial - gt_initial)

    # Tolerance for floating point precision
    tolerance = 1e-6

    # Log the check results
    logging.info(f"=== SEGMENT {seg_idx} SANITY CHECK ===")
    logging.info(f"GT initial position: {gt_initial}")
    logging.info(f"Pred initial position: {pred_initial}")
    logging.info(f"Initial position difference: {initial_diff:.8f}")
    logging.info(f"Tolerance: {tolerance}")
    logging.info(f"Alignment correct: {initial_diff <= tolerance}")
    logging.info(f"Segment length: {segment.get('segment_length', 'N/A')}m")
    logging.info(f"Number of points: {len(pos_pred)}")
    logging.info("=" * 50)

    return initial_diff <= tolerance


def create_segment_plot(segment, segment_metric, outdir, seg_idx):
    """
    Create plots for a single trajectory segment.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        outdir: Output directory
        seg_idx: Segment index
    """
    import matplotlib.pyplot as plt

    # Create segment directory
    segment_outdir = osp.join(outdir, f"segment_{seg_idx:03d}")
    if not osp.exists(segment_outdir):
        os.makedirs(segment_outdir)

    # SANITY CHECK: Verify that inference starts from ground truth initial position
    alignment_correct = sanity_check_segment_alignment(segment, seg_idx)

    # Extract data
    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]
    ts = segment["ts"]

    # FIX: Align segment to start from origin (0,0,0) for consistent visualization
    # This makes all segments start from the same reference point for easy comparison
    gt_initial = pos_gt[0]
    pred_initial = pos_pred[0]

    # Align both trajectories to start from origin (0,0,0)
    pos_pred_aligned = pos_pred - gt_initial  # Start from origin
    pos_gt_aligned = pos_gt - gt_initial  # Start from origin

    # Create 2D trajectory plot
    plt.figure(figsize=(12, 8))

    # Main trajectory plot
    plt.subplot(2, 2, 1)
    plt.plot(
        pos_pred_aligned[:, 0],
        pos_pred_aligned[:, 1],
        "b-",
        linewidth=2,
        label="Predicted",
    )
    plt.plot(
        pos_gt_aligned[:, 0],
        pos_gt_aligned[:, 1],
        "r-",
        linewidth=2,
        label="Ground Truth",
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Segment {seg_idx} - 2D Trajectory (Origin Aligned)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # Mark the start point
    plt.plot(0, 0, "ko", markersize=8, label="Start (0,0)")

    # 3D trajectory plot
    ax = plt.subplot(2, 2, 2, projection="3d")
    ax.plot(
        pos_pred_aligned[:, 0],
        pos_pred_aligned[:, 1],
        pos_pred_aligned[:, 2],
        "b-",
        linewidth=2,
        label="Predicted",
    )
    ax.plot(
        pos_gt_aligned[:, 0],
        pos_gt_aligned[:, 1],
        pos_gt_aligned[:, 2],
        "r-",
        linewidth=2,
        label="Ground Truth",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Segment {seg_idx} - 3D Trajectory (Origin Aligned)")
    ax.legend()
    ax.grid(True)

    # Mark the start point in 3D
    ax.scatter([0], [0], [0], c="black", s=100, marker="o", label="Start (0,0,0)")

    # Position error over time (using aligned positions for proper error calculation)
    plt.subplot(2, 2, 3)
    pos_error = np.linalg.norm(pos_pred_aligned - pos_gt_aligned, axis=1)
    plt.plot(ts, pos_error, "g-", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title(f"Segment {seg_idx} - Position Error (Origin Aligned)")
    plt.grid(True)

    # Metrics summary
    plt.subplot(2, 2, 4)
    alignment_status = "✅ CORRECT" if alignment_correct else "❌ INCORRECT"
    metrics_text = f"""Segment {seg_idx} Metrics (Origin Aligned):

    ATE: {segment_metric['ate']:.4f} m
    T_RTE: {segment_metric['t_rte']:.4f} m
    D_RTE: {segment_metric['d_rte']:.4f} m
    P_RMSE: {segment_metric['P_RMSE']:.4f} m
    V_RMSE: {segment_metric['V_RMSE']:.4f} m/s
    Segment Length: {segment_metric['segment_length']:.2f} m
    Points: {segment_metric['end_idx'] - segment_metric['start_idx'] + 1}

    ORIGINAL POSITIONS:
    GT Start: {gt_initial}
    Pred Start: {pred_initial}

    ALIGNED POSITIONS:
    GT Start: (0, 0, 0)
    Pred Start: {pred_initial - gt_initial}

    Alignment: {alignment_status}
    Status: {'Inference starts from GT position' if alignment_correct else 'WARNING: Check inference logic'}"""

    plt.text(
        0.1,
        0.5,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
    )
    plt.axis("off")
    plt.title(f"Segment {seg_idx} - Metrics Summary")

    plt.tight_layout()
    plt.savefig(
        osp.join(segment_outdir, f"segment_{seg_idx:03d}_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create additional detailed plots
    create_detailed_segment_plots(segment, segment_metric, segment_outdir, seg_idx)


def create_detailed_segment_plots(segment, segment_metric, segment_outdir, seg_idx):
    """
    Create detailed plots for a segment including velocity and acceleration.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        segment_outdir: Segment output directory
        seg_idx: Segment index
    """
    import matplotlib.pyplot as plt

    pos_pred = segment["pos_pred"]
    pos_gt = segment["pos_gt"]
    ts = segment["ts"]

    # Align segment to start from origin (0,0,0) for consistent visualization (same as in create_segment_plot)
    gt_initial = pos_gt[0]
    pred_initial = pos_pred[0]
    pos_pred_aligned = pos_pred - gt_initial  # Start from origin
    pos_gt_aligned = pos_gt - gt_initial  # Start from origin

    # Use direct velocity predictions if available, otherwise calculate from position
    if (
        "vel_pred" in segment
        and "vel_gt" in segment
        and segment["vel_pred"] is not None
    ):
        # Use direct velocity predictions from model
        vel_pred = segment["vel_pred"]
        vel_gt = segment["vel_gt"]
        vel_ts = ts  # Use same timestamps as position
        logging.info(
            f"Using direct velocity predictions - vel_pred shape: {vel_pred.shape}"
        )
    else:
        # Fallback: Calculate velocities (simple finite difference) using aligned positions
        dt = np.diff(ts)
        vel_pred = np.diff(pos_pred_aligned, axis=0) / dt[:, np.newaxis]
        vel_gt = np.diff(pos_gt_aligned, axis=0) / dt[:, np.newaxis]
        vel_ts = ts[:-1]  # Time stamps for velocity
        logging.info(
            f"Using finite difference velocity calculation - vel_pred shape: {vel_pred.shape}"
        )

    # Calculate accelerations
    if (vel_pred is not None and vel_gt is not None and 
        len(vel_pred) > 1 and len(vel_gt) > 1 and 
        vel_pred.ndim > 0 and vel_gt.ndim > 0):
        dt_vel = np.diff(vel_ts)
        acc_pred = np.diff(vel_pred, axis=0) / dt_vel[:, np.newaxis]
        acc_gt = np.diff(vel_gt, axis=0) / dt_vel[:, np.newaxis]
        acc_ts = vel_ts[:-1]
    else:
        acc_pred = acc_gt = acc_ts = np.array([])

    # Velocity plot
    plt.figure(figsize=(15, 10))

    # Velocity components
    plt.subplot(3, 3, 1)
    plt.plot(vel_ts, vel_pred[:, 0], "b-", label="Pred X")
    plt.plot(vel_ts, vel_gt[:, 0], "r-", label="GT X")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity X (m/s)")
    plt.title("Velocity X Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(vel_ts, vel_pred[:, 1], "b-", label="Pred Y")
    plt.plot(vel_ts, vel_gt[:, 1], "r-", label="GT Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Y (m/s)")
    plt.title("Velocity Y Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(vel_ts, vel_pred[:, 2], "b-", label="Pred Z")
    plt.plot(vel_ts, vel_gt[:, 2], "r-", label="GT Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Z (m/s)")
    plt.title("Velocity Z Component")
    plt.legend()
    plt.grid(True)

    # Velocity magnitude
    plt.subplot(3, 3, 4)
    vel_pred_mag = np.linalg.norm(vel_pred, axis=1)
    vel_gt_mag = np.linalg.norm(vel_gt, axis=1)
    plt.plot(vel_ts, vel_pred_mag, "b-", label="Predicted")
    plt.plot(vel_ts, vel_gt_mag, "r-", label="Ground Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Magnitude (m/s)")
    plt.title("Velocity Magnitude")
    plt.legend()
    plt.grid(True)

    # Position components
    plt.subplot(3, 3, 5)
    plt.plot(ts, pos_pred[:, 0], "b-", label="Pred X")
    plt.plot(ts, pos_gt[:, 0], "r-", label="GT X")
    plt.xlabel("Time (s)")
    plt.ylabel("Position X (m)")
    plt.title("Position X Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(ts, pos_pred[:, 1], "b-", label="Pred Y")
    plt.plot(ts, pos_gt[:, 1], "r-", label="GT Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Y (m)")
    plt.title("Position Y Component")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(ts, pos_pred[:, 2], "b-", label="Pred Z")
    plt.plot(ts, pos_gt[:, 2], "r-", label="GT Z")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Z (m)")
    plt.title("Position Z Component")
    plt.legend()
    plt.grid(True)

    # Error analysis
    plt.subplot(3, 3, 8)
    pos_error_components = np.abs(pos_pred - pos_gt)
    plt.plot(ts, pos_error_components[:, 0], "g-", label="X Error")
    plt.plot(ts, pos_error_components[:, 1], "m-", label="Y Error")
    plt.plot(ts, pos_error_components[:, 2], "c-", label="Z Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.title("Position Error Components")
    plt.legend()
    plt.grid(True)

    # Cumulative distance
    plt.subplot(3, 3, 9)
    cum_dist_pred = np.cumsum(np.linalg.norm(np.diff(pos_pred, axis=0), axis=1))
    cum_dist_gt = np.cumsum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
    plt.plot(ts[1:], cum_dist_pred, "b-", label="Predicted")
    plt.plot(ts[1:], cum_dist_gt, "r-", label="Ground Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative Distance (m)")
    plt.title("Cumulative Distance")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(
        osp.join(segment_outdir, f"segment_{seg_idx:03d}_detailed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_segment_data(segment, segment_metric, outdir, seg_idx):
    """
    Save segment trajectory and metrics to files.

    Args:
        segment: Trajectory segment data
        segment_metric: Segment metrics
        outdir: Output directory
        seg_idx: Segment index
    """
    # Create segment directory
    segment_outdir = osp.join(outdir, f"segment_{seg_idx:03d}")
    if not osp.exists(segment_outdir):
        os.makedirs(segment_outdir)

    # Save segment trajectory data
    segment_trajectory_data = np.concatenate(
        [
            segment["ts"].reshape(-1, 1),
            segment["pos_pred"],
            segment["pos_gt"],
            segment["cov_pred"],
        ],
        axis=1,
    )
    segment_traj_file = osp.join(segment_outdir, "trajectory.txt")
    np.savetxt(segment_traj_file, segment_trajectory_data, delimiter=",")

    # Save segment metrics
    segment_metrics_file = osp.join(segment_outdir, "metrics.json")
    with open(segment_metrics_file, "w") as f:
        json.dump(segment_metric, f, indent=1)

    # Create plots for this segment
    create_segment_plot(segment, segment_metric, outdir, seg_idx)


def compute_aggregated_metrics(segment_metrics):
    """
    Compute aggregated metrics from segment metrics.

    Args:
        segment_metrics: List of segment metrics

    Returns:
        Dictionary of aggregated metrics
    """
    if not segment_metrics:
        return {}

    metrics_keys = [
        "ate",
        "t_rte",
        "d_rte",
        "ATE",
        "AVE",
        "P_RMSE",
        "V_RMSE",
        "X_ATE",
        "Y_ATE",
        "Z_ATE",
        "X_AVE",
        "Y_AVE",
        "Z_AVE",
    ]

    aggregated = {}
    for key in metrics_keys:
        values = [seg[key] for seg in segment_metrics]
        aggregated[f"avg_{key}"] = np.mean(values)

    return aggregated


def save_full_trajectory_data(traj_attr_dict, outdir, epoch_num, plot_dict):
    """
    Save full trajectory data and metrics.

    Args:
        traj_attr_dict: Full trajectory data
        outdir: Output directory
        epoch_num: Epoch number
        plot_dict: Plot dictionary
    """
    # Save full trajectory data
    outfile = osp.join(outdir, "trajectory.txt")
    trajectory_data = np.concatenate(
        [
            traj_attr_dict["ts"].reshape(-1, 1),
            traj_attr_dict["pos_pred"],
            traj_attr_dict["pos_gt"],
            traj_attr_dict["cov_pred"],
        ],
        axis=1,
    )
    np.savetxt(outfile, trajectory_data, delimiter=",")

    # Save network outputs
    outfile_net = osp.join(outdir, "net_outputs.txt")
    est_pose_file = osp.join(outdir, f"est_pose_{epoch_num}.txt")
    gt_pose_file = osp.join(outdir, f"gt_pose_{epoch_num}.txt")

    net_outputs_data = np.concatenate(
        [
            plot_dict["pred_ts"].reshape(-1, 1),
            plot_dict["preds"],
            plot_dict["targets"],
            plot_dict["pred_sigmas"],
        ],
        axis=1,
    )

    est_pose = plot_dict["pos_pred"]
    gt_pose = plot_dict["pos_gt"]
    np.savetxt(gt_pose_file, gt_pose, delimiter=",")
    np.savetxt(est_pose_file, est_pose, delimiter=",")


def compute_full_trajectory_metrics(traj_attr_dict, cfg):
    """
    Compute metrics for the full trajectory.

    Args:
        traj_attr_dict: Full trajectory data
        cfg: Configuration dictionary

    Returns:
        Dictionary of full trajectory metrics
    """
    ate, t_rte, d_rte = compute_ate_rte(
        traj_attr_dict["pos_pred"],
        traj_attr_dict["pos_gt"],
        int(cfg["data"]["imu_freq"] * 1),
    )

    ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE = (
        compute_accruacy_metrics(
            traj_attr_dict,
            int(cfg["data"]["imu_freq"] * 1),
            cfg["data"]["use_local_coord"],
        )
    )

    return {
        "ate": ate,
        "t_rte": t_rte,
        "d_rte": d_rte,
        "ATE": ATE,
        "AVE": AVE,
        "P_RMSE": P_RMSE,
        "V_RMSE": V_RMSE,
        "X_ATE": X_ATE,
        "Y_ATE": Y_ATE,
        "Z_ATE": Z_ATE,
        "X_AVE": X_AVE,
        "Y_AVE": Y_AVE,
        "Z_AVE": Z_AVE,
    }


def save_metrics_files(all_metrics, outdir, data_name, segment_metrics):
    """
    Save all metrics to JSON files.

    Args:
        all_metrics: All metrics dictionary
        outdir: Output directory
        data_name: Data name
        segment_metrics: List of segment metrics
    """
    # Save main metrics
    with open(outdir + "/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=1)

    # Save segment metrics summary
    segment_summary = {
        "trajectory": data_name,
        "num_segments": len(segment_metrics),
        "segment_metrics": segment_metrics,
    }
    with open(outdir + "/segment_metrics_summary.json", "w") as f:
        json.dump(segment_summary, f, indent=1)


def compute_overall_statistics(all_trajectory_results, segment_metrics_all):
    """
    Compute overall statistics across all trajectories and segments.

    Args:
        all_trajectory_results: List of all trajectory results
        segment_metrics_all: List of all segment metrics

    Returns:
        Dictionary of overall statistics
    """
    # Extract aggregated metrics from all trajectories
    metrics_keys = [
        "avg_ate",
        "avg_t_rte",
        "avg_d_rte",
        "avg_ATE",
        "avg_AVE",
        "avg_P_RMSE",
        "avg_V_RMSE",
        "avg_X_ATE",
        "avg_Y_ATE",
        "avg_Z_ATE",
        "avg_X_AVE",
        "avg_Y_AVE",
        "avg_Z_AVE",
    ]

    overall_stats = {"total_segments": len(segment_metrics_all)}

    for key in metrics_keys:
        values = [traj[key] for traj in all_trajectory_results if key in traj]
        if values:
            overall_stats[key] = float(np.mean(values))

    # Add segment-wise statistics
    if segment_metrics_all:
        overall_stats["segment_statistics"] = {
            "total_segments": len(segment_metrics_all),
            "avg_segment_length": float(
                np.mean([seg["segment_length"] for seg in segment_metrics_all])
            ),
            "segment_ate_stats": {
                "mean": float(np.mean([seg["ate"] for seg in segment_metrics_all])),
                "std": float(np.std([seg["ate"] for seg in segment_metrics_all])),
                "min": float(np.min([seg["ate"] for seg in segment_metrics_all])),
                "max": float(np.max([seg["ate"] for seg in segment_metrics_all])),
            },
            "segment_rmse_stats": {
                "mean": float(np.mean([seg["P_RMSE"] for seg in segment_metrics_all])),
                "std": float(np.std([seg["P_RMSE"] for seg in segment_metrics_all])),
                "min": float(np.min([seg["P_RMSE"] for seg in segment_metrics_all])),
                "max": float(np.max([seg["P_RMSE"] for seg in segment_metrics_all])),
            },
        }

    return overall_stats


def create_3d_segments_summary(all_trajectory_results, out_dir):
    """
    Create a 3D summary plot showing all segments aligned to origin for easy comparison.
    Also includes the full trajectory with ground truth position correction.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    # Create figure with 3D projection
    fig = plt.figure(figsize=(24, 18))

    # Main 3D plot showing all segments
    ax1 = fig.add_subplot(3, 4, 1, projection="3d")

    # Collect all segment data and align to origin
    all_segments_data = []
    colors = cm.tab20(np.linspace(0, 1, 20))  # Use 20 different colors

    for traj_idx, traj_result in enumerate(all_trajectory_results):
        for seg_idx, seg_metric in enumerate(traj_result["segment_metrics"]):
            # Load segment data from saved files
            segment_dir = osp.join(out_dir, f"segment_{seg_idx:03d}")
            trajectory_file = osp.join(segment_dir, "trajectory.txt")

            if osp.exists(trajectory_file):
                # Load trajectory data: [ts, pos_pred_x, pos_pred_y, pos_pred_z, pos_gt_x, pos_gt_y, pos_gt_z, cov_pred_x, cov_pred_y, cov_pred_z]
                traj_data = np.loadtxt(trajectory_file, delimiter=",")

                # Extract positions
                pos_pred = traj_data[:, 1:4]  # predicted positions
                pos_gt = traj_data[:, 4:7]  # ground truth positions

                # Align to origin
                gt_initial = pos_gt[0]
                pos_pred_aligned = pos_pred - gt_initial
                pos_gt_aligned = pos_gt - gt_initial

                all_segments_data.append(
                    {
                        "traj_idx": traj_idx,
                        "seg_idx": seg_idx,
                        "pos_pred": pos_pred_aligned,
                        "pos_gt": pos_gt_aligned,
                        "ate": seg_metric["ate"],
                        "color": colors[seg_idx % len(colors)],
                    }
                )

    # Plot all segments in 3D
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax1.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 1],
            seg_data["pos_gt"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            label=f'GT Seg {seg_data["seg_idx"]}' if seg_data["seg_idx"] < 5 else "",
        )

        # Plot predicted (dashed line)
        ax1.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 1],
            seg_data["pos_pred"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
            label=f'Pred Seg {seg_data["seg_idx"]}' if seg_data["seg_idx"] < 5 else "",
        )

        # Mark start points
        ax1.scatter([0], [0], [0], c="black", s=50, marker="o", alpha=0.8)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(
        "All Segments Aligned to Origin (0,0,0)\nGround Truth: Solid, Predicted: Dashed"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Full trajectory with drift correction
    ax2 = fig.add_subplot(3, 4, 2, projection="3d")

    # Load full trajectory data
    full_trajectory_file = osp.join(out_dir, "trajectory.txt")
    full_pos_pred = None
    full_pos_gt = None
    full_gt_initial = None
    full_pred_initial = None
    full_pos_pred_corrected = None

    if osp.exists(full_trajectory_file):
        full_traj_data = np.loadtxt(full_trajectory_file, delimiter=",")

        # Extract full trajectory positions
        full_pos_pred = full_traj_data[:, 1:4]  # predicted positions
        full_pos_gt = full_traj_data[:, 4:7]  # ground truth positions

        # Apply drift correction to full trajectory (same as in process_single_trajectory)
        full_gt_initial = full_pos_gt[0]
        full_pred_initial = full_pos_pred[0]
        full_pos_pred_corrected = full_pos_pred - full_pred_initial + full_gt_initial

        # Plot full trajectory
        ax2.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            full_pos_gt[:, 2],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax2.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            full_pos_pred_corrected[:, 2],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax2.scatter(
            [full_gt_initial[0]],
            [full_gt_initial[1]],
            [full_gt_initial[2]],
            c="green",
            s=100,
            marker="o",
            label="Start",
        )
        ax2.scatter(
            [full_pos_gt[-1, 0]],
            [full_pos_gt[-1, 1]],
            [full_pos_gt[-1, 2]],
            c="red",
            s=100,
            marker="s",
            label="End",
        )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title(
        "Full Trajectory with Drift Correction\nGT: Red Solid, Pred: Blue Dashed"
    )
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: ATE vs Segment Index
    ax3 = fig.add_subplot(3, 4, 3)

    if all_segments_data:  # Check if segments exist
        segment_indices = [seg_data["seg_idx"] for seg_data in all_segments_data]
        ates = [seg_data["ate"] for seg_data in all_segments_data]
        colors_ate = [seg_data["color"] for seg_data in all_segments_data]

        scatter = ax3.scatter(segment_indices, ates, c=colors_ate, alpha=0.7, s=50)
        ax3.set_xlabel("Segment Index")
        ax3.set_ylabel("ATE (m)")
        ax3.set_title("ATE vs Segment Index")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No segments found",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("ATE vs Segment Index (No Data)")

    # Plot 4: Segment length distribution
    ax4 = fig.add_subplot(3, 4, 4)
    segment_lengths = [
        seg_metric["segment_length"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]

    if segment_lengths:  # Check if segment lengths exist
        ax4.hist(
            segment_lengths, bins=15, alpha=0.7, color="lightgreen", edgecolor="black"
        )
        ax4.set_xlabel("Segment Length (m)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Segment Length Distribution")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No segments found",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Segment Length Distribution (No Data)")

    # Plot 5: 2D projection (X-Y) of all segments
    ax5 = fig.add_subplot(3, 4, 5)
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax5.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 1],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

        # Plot predicted (dashed line)
        ax5.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 1],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
        )

    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_title("2D Projection (X-Y) of All Segments")
    ax5.grid(True, alpha=0.3)
    ax5.axis("equal")

    # Mark origin
    ax5.plot(0, 0, "ko", markersize=8, label="Origin (0,0)")
    ax5.legend()

    # Plot 6: 2D projection (X-Z) of all segments
    ax6 = fig.add_subplot(3, 4, 6)
    for seg_data in all_segments_data:
        color = seg_data["color"]
        alpha = 0.7

        # Plot ground truth (solid line)
        ax6.plot(
            seg_data["pos_gt"][:, 0],
            seg_data["pos_gt"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
        )

        # Plot predicted (dashed line)
        ax6.plot(
            seg_data["pos_pred"][:, 0],
            seg_data["pos_pred"][:, 2],
            color=color,
            linewidth=2,
            alpha=alpha,
            linestyle="--",
        )

    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Z (m)")
    ax6.set_title("2D Projection (X-Z) of All Segments")
    ax6.grid(True, alpha=0.3)

    # Mark origin
    ax6.plot(0, 0, "ko", markersize=8, label="Origin (0,0)")
    ax6.legend()

    # Plot 7: 2D projection (X-Y) of full trajectory
    ax7 = fig.add_subplot(3, 4, 7)
    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        ax7.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax7.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax7.plot(
            full_gt_initial[0], full_gt_initial[1], "go", markersize=10, label="Start"
        )
        ax7.plot(
            full_pos_gt[-1, 0], full_pos_gt[-1, 1], "rs", markersize=10, label="End"
        )

    ax7.set_xlabel("X (m)")
    ax7.set_ylabel("Y (m)")
    ax7.set_title("2D Projection (X-Y) of Full Trajectory")
    ax7.grid(True, alpha=0.3)
    ax7.axis("equal")
    ax7.legend()

    # Plot 8: 2D projection (X-Z) of full trajectory
    ax8 = fig.add_subplot(3, 4, 8)
    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        ax8.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 2],
            color="red",
            linewidth=3,
            label="Full GT Trajectory",
        )
        ax8.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 2],
            color="blue",
            linewidth=3,
            linestyle="--",
            label="Full Pred Trajectory (Drift Corrected)",
        )

        # Mark start and end points
        ax8.plot(
            full_gt_initial[0], full_gt_initial[2], "go", markersize=10, label="Start"
        )
        ax8.plot(
            full_pos_gt[-1, 0], full_pos_gt[-1, 2], "rs", markersize=10, label="End"
        )

    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Z (m)")
    ax8.set_title("2D Projection (X-Z) of Full Trajectory")
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # Plot 9: Comparison of full trajectory vs segments
    ax9 = fig.add_subplot(3, 4, 9)

    # Calculate full trajectory ATE
    if (
        full_pos_gt is not None
        and full_pos_pred_corrected is not None
        and all_segments_data
    ):
        full_ate = np.mean(
            np.linalg.norm(full_pos_pred_corrected - full_pos_gt, axis=1)
        )
        segment_avg_ate = np.mean(ates)

        comparison_data = ["Full Trajectory", "Average Segments"]
        comparison_values = [full_ate, segment_avg_ate]
        colors_comp = ["red", "blue"]

        bars = ax9.bar(comparison_data, comparison_values, color=colors_comp, alpha=0.7)
        ax9.set_ylabel("ATE (m)")
        ax9.set_title("Full Trajectory vs Segment Average ATE")
        ax9.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, comparison_values):
            ax9.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax9.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax9.transAxes,
        )
        ax9.set_title("Full Trajectory vs Segment Average ATE (No Data)")

    # Plot 10: Trajectory length comparison
    ax10 = fig.add_subplot(3, 4, 10)
    if (
        full_pos_gt is not None
        and full_pos_pred_corrected is not None
        and segment_lengths
    ):
        # Calculate trajectory lengths
        full_gt_length = np.sum(np.linalg.norm(np.diff(full_pos_gt, axis=0), axis=1))
        full_pred_length = np.sum(
            np.linalg.norm(np.diff(full_pos_pred_corrected, axis=0), axis=1)
        )
        segment_avg_length = np.mean(segment_lengths)

        length_data = ["Full GT", "Full Pred", "Avg Segment"]
        length_values = [full_gt_length, full_pred_length, segment_avg_length]
        colors_length = ["red", "blue", "green"]

        bars = ax10.bar(length_data, length_values, color=colors_length, alpha=0.7)
        ax10.set_ylabel("Length (m)")
        ax10.set_title("Trajectory Length Comparison")
        ax10.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, length_values):
            ax10.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax10.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax10.transAxes,
        )
        ax10.set_title("Trajectory Length Comparison (No Data)")

    # Plot 11: Summary statistics
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis("off")

    # Calculate summary statistics
    total_segments = len(all_segments_data)

    if all_segments_data:  # Check if segments exist
        avg_ate = np.mean(ates)
        std_ate = np.std(ates)
        min_ate = np.min(ates)
        max_ate = np.max(ates)
    else:
        avg_ate = "N/A"
        std_ate = "N/A"
        min_ate = "N/A"
        max_ate = "N/A"

    if full_pos_gt is not None and full_pos_pred_corrected is not None:
        full_ate = np.mean(
            np.linalg.norm(full_pos_pred_corrected - full_pos_gt, axis=1)
        )
        full_gt_length = np.sum(np.linalg.norm(np.diff(full_pos_gt, axis=0), axis=1))
    else:
        full_ate = "N/A"
        full_gt_length = "N/A"

    # Format ATE values based on their type
    if isinstance(avg_ate, (int, float)):
        avg_ate_str = f"{avg_ate:.4f}"
    else:
        avg_ate_str = str(avg_ate)

    if isinstance(std_ate, (int, float)):
        std_ate_str = f"{std_ate:.4f}"
    else:
        std_ate_str = str(std_ate)

    if isinstance(min_ate, (int, float)):
        min_ate_str = f"{min_ate:.4f}"
    else:
        min_ate_str = str(min_ate)

    if isinstance(max_ate, (int, float)):
        max_ate_str = f"{max_ate:.4f}"
    else:
        max_ate_str = str(max_ate)

    summary_text = f"""COMPREHENSIVE SUMMARY

SEGMENTS:
Total Segments: {total_segments}
Average ATE: {avg_ate_str} m
ATE Std Dev: {std_ate_str} m
Min ATE: {min_ate_str} m
Max ATE: {max_ate_str} m

FULL TRAJECTORY:
Full Trajectory ATE: {full_ate} m
Full Trajectory Length: {full_gt_length} m

VISUALIZATION:
- Segments aligned to origin (0,0,0)
- Full trajectory with drift correction
- Ground Truth: Solid lines
- Predicted: Dashed lines"""

    ax11.text(
        0.1,
        0.5,
        summary_text,
        transform=ax11.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    # Plot 12: Drift correction visualization
    ax12 = fig.add_subplot(3, 4, 12)
    if (
        full_pos_gt is not None
        and full_pos_pred is not None
        and full_pos_pred_corrected is not None
    ):
        # Show original vs corrected prediction
        ax12.plot(
            full_pos_gt[:, 0],
            full_pos_gt[:, 1],
            color="red",
            linewidth=3,
            label="Ground Truth",
        )
        ax12.plot(
            full_pos_pred[:, 0],
            full_pos_pred[:, 1],
            color="orange",
            linewidth=2,
            linestyle=":",
            label="Original Prediction",
        )
        ax12.plot(
            full_pos_pred_corrected[:, 0],
            full_pos_pred_corrected[:, 1],
            color="blue",
            linewidth=2,
            linestyle="--",
            label="Drift Corrected Prediction",
        )

        # Mark start points
        ax12.plot(
            full_gt_initial[0], full_gt_initial[1], "go", markersize=10, label="Start"
        )
        ax12.plot(
            full_pos_pred[0, 0],
            full_pos_pred[0, 1],
            "mo",
            markersize=8,
            label="Original Start",
        )
        ax12.plot(
            full_pos_pred_corrected[0, 0],
            full_pos_pred_corrected[0, 1],
            "co",
            markersize=8,
            label="Corrected Start",
        )

    ax12.set_xlabel("X (m)")
    ax12.set_ylabel("Y (m)")
    ax12.set_title("Drift Correction Effect")
    ax12.grid(True, alpha=0.3)
    ax12.axis("equal")
    ax12.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "3d_comprehensive_summary.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def display_rich_metrics_tables(
    all_trajectory_results, overall_stats, segment_metrics_all
):
    """
    Display comprehensive metrics using rich tables.

    Args:
        all_trajectory_results: List of all trajectory results
        overall_stats: Overall statistics dictionary
        segment_metrics_all: List of all segment metrics
    """
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # 1. Overall Summary Table
    summary_table = Table(
        title="📊 Overall Test Results Summary",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    summary_table.add_column("Metric", style="cyan", no_wrap=True)
    summary_table.add_column("Value", style="green", justify="right")
    summary_table.add_column("Description", style="yellow")

    summary_table.add_row(
        "Total Trajectories",
        str(len(all_trajectory_results)),
        "Number of test trajectories",
    )
    summary_table.add_row(
        "Total Segments",
        str(overall_stats.get("total_segments", 0)),
        "Total number of 5m segments",
    )

    if "segment_statistics" in overall_stats:
        seg_stats = overall_stats["segment_statistics"]
        summary_table.add_row(
            "Avg Segment Length",
            f"{seg_stats.get('avg_segment_length', 0):.2f} m",
            "Average length of segments",
        )
        summary_table.add_row(
            "Avg Segment ATE",
            f"{seg_stats['segment_ate_stats']['mean']:.4f} m",
            "Average ATE across all segments",
        )
        summary_table.add_row(
            "Segment ATE Std",
            f"{seg_stats['segment_ate_stats']['std']:.4f} m",
            "Standard deviation of segment ATE",
        )
        summary_table.add_row(
            "Segment ATE Range",
            f"{seg_stats['segment_ate_stats']['min']:.4f} - {seg_stats['segment_ate_stats']['max']:.4f} m",
            "Min-Max ATE across segments",
        )

    console.print(summary_table)
    console.print()

    # 2. Trajectory-wise Metrics Table
    traj_table = Table(
        title="🚀 Individual Trajectory Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold blue",
    )
    traj_table.add_column("Trajectory", style="cyan", no_wrap=True)
    traj_table.add_column("Data", style="magenta")
    traj_table.add_column("Segments", style="green", justify="right")
    traj_table.add_column("Avg ATE (m)", style="yellow", justify="right")
    traj_table.add_column("Avg RMSE (m)", style="red", justify="right")
    traj_table.add_column("T_RTE (m)", style="blue", justify="right")
    traj_table.add_column("D_RTE (m)", style="cyan", justify="right")

    for i, traj_result in enumerate(all_trajectory_results):
        traj_table.add_row(
            f"Traj_{i+1}",
            traj_result.get("data", "N/A"),
            str(traj_result.get("num_segments", 0)),
            f"{traj_result.get('avg_ate', 0):.4f}",
            f"{traj_result.get('avg_P_RMSE', 0):.4f}",
            f"{traj_result.get('avg_t_rte', 0):.4f}",
            f"{traj_result.get('avg_d_rte', 0):.4f}",
        )

    console.print(traj_table)
    console.print()

    # 3. Segment Statistics Table
    if segment_metrics_all:
        seg_table = Table(
            title="📏 Segment-level Statistics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold green",
        )
        seg_table.add_column("Metric", style="cyan", no_wrap=True)
        seg_table.add_column("Mean", style="green", justify="right")
        seg_table.add_column("Std", style="yellow", justify="right")
        seg_table.add_column("Min", style="red", justify="right")
        seg_table.add_column("Max", style="blue", justify="right")

        # Calculate statistics for key metrics
        metrics_to_show = ["ate", "P_RMSE", "V_RMSE", "t_rte", "d_rte", "ATE", "AVE"]
        for metric in metrics_to_show:
            values = [
                seg.get(metric, 0) for seg in segment_metrics_all if metric in seg
            ]
            if values:
                seg_table.add_row(
                    metric.upper(),
                    f"{np.mean(values):.4f}",
                    f"{np.std(values):.4f}",
                    f"{np.min(values):.4f}",
                    f"{np.max(values):.4f}",
                )

        console.print(seg_table)
        console.print()

    # 4. Full Trajectory Metrics Table
    full_traj_table = Table(
        title="🎯 Full Trajectory Metrics",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold red",
    )
    full_traj_table.add_column("Trajectory", style="cyan", no_wrap=True)
    full_traj_table.add_column("ATE (m)", style="green", justify="right")
    full_traj_table.add_column("T_RTE (m)", style="yellow", justify="right")
    full_traj_table.add_column("D_RTE (m)", style="red", justify="right")
    full_traj_table.add_column("P_RMSE (m)", style="blue", justify="right")
    full_traj_table.add_column("V_RMSE (m/s)", style="magenta", justify="right")

    for i, traj_result in enumerate(all_trajectory_results):
        full_traj = traj_result.get("full_trajectory", {})
        full_traj_table.add_row(
            f"Traj_{i+1}",
            f"{full_traj.get('ate', 0):.4f}",
            f"{full_traj.get('t_rte', 0):.4f}",
            f"{full_traj.get('d_rte', 0):.4f}",
            f"{full_traj.get('P_RMSE', 0):.4f}",
            f"{full_traj.get('V_RMSE', 0):.4f}",
        )

    console.print(full_traj_table)
    console.print()


def save_comprehensive_csv_files(
    all_trajectory_results, overall_stats, segment_metrics_all, out_dir
):
    """
    Save comprehensive CSV files with all metrics organized by category.

    Args:
        all_trajectory_results: List of all trajectory results
        overall_stats: Overall statistics dictionary
        segment_metrics_all: List of all segment metrics
        out_dir: Output directory
    """

    # 1. Overall Summary CSV
    summary_file = osp.join(out_dir, "overall_summary.csv")
    with open(summary_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Category", "Metric", "Value", "Description"])

        # Basic statistics
        writer.writerow(
            [
                "Basic",
                "Total Trajectories",
                len(all_trajectory_results),
                "Number of test trajectories",
            ]
        )
        writer.writerow(
            [
                "Basic",
                "Total Segments",
                overall_stats.get("total_segments", 0),
                "Total number of 5m segments",
            ]
        )

        # Overall aggregated metrics
        for key, value in overall_stats.items():
            if key not in ["total_segments", "segment_statistics"]:
                writer.writerow(["Overall", key, f"{value:.6f}", f"Overall {key}"])

        # Segment statistics
        if "segment_statistics" in overall_stats:
            seg_stats = overall_stats["segment_statistics"]
            writer.writerow(
                [
                    "Segments",
                    "Average Segment Length",
                    f"{seg_stats.get('avg_segment_length', 0):.4f}",
                    "Average length of segments",
                ]
            )

            ate_stats = seg_stats.get("segment_ate_stats", {})
            writer.writerow(
                [
                    "Segments",
                    "ATE Mean",
                    f"{ate_stats.get('mean', 0):.6f}",
                    "Average ATE across all segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Std",
                    f"{ate_stats.get('std', 0):.6f}",
                    "Standard deviation of segment ATE",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Min",
                    f"{ate_stats.get('min', 0):.6f}",
                    "Minimum ATE across segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "ATE Max",
                    f"{ate_stats.get('max', 0):.6f}",
                    "Maximum ATE across segments",
                ]
            )

            rmse_stats = seg_stats.get("segment_rmse_stats", {})
            writer.writerow(
                [
                    "Segments",
                    "RMSE Mean",
                    f"{rmse_stats.get('mean', 0):.6f}",
                    "Average RMSE across all segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Std",
                    f"{rmse_stats.get('std', 0):.6f}",
                    "Standard deviation of segment RMSE",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Min",
                    f"{rmse_stats.get('min', 0):.6f}",
                    "Minimum RMSE across segments",
                ]
            )
            writer.writerow(
                [
                    "Segments",
                    "RMSE Max",
                    f"{rmse_stats.get('max', 0):.6f}",
                    "Maximum RMSE across segments",
                ]
            )

    # 2. Trajectory-wise Metrics CSV
    traj_file = osp.join(out_dir, "trajectory_metrics.csv")
    with open(traj_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Trajectory_ID",
                "Data_Name",
                "Robot_Type",
                "Num_Segments",
                "Avg_ATE",
                "Avg_T_RTE",
                "Avg_D_RTE",
                "Avg_ATE_metric",
                "Avg_AVE",
                "Avg_P_RMSE",
                "Avg_V_RMSE",
                "Avg_X_ATE",
                "Avg_Y_ATE",
                "Avg_Z_ATE",
                "Avg_X_AVE",
                "Avg_Y_AVE",
                "Avg_Z_AVE",
                "Full_ATE",
                "Full_T_RTE",
                "Full_D_RTE",
                "Full_ATE_metric",
                "Full_AVE",
                "Full_P_RMSE",
                "Full_V_RMSE",
                "Full_X_ATE",
                "Full_Y_ATE",
                "Full_Z_ATE",
                "Full_X_AVE",
                "Full_Y_AVE",
                "Full_Z_AVE",
            ]
        )

        for i, traj_result in enumerate(all_trajectory_results):
            full_traj = traj_result.get("full_trajectory", {})
            row = [
                f"Traj_{i+1}",
                traj_result.get("data", "N/A"),
                traj_result.get("robot_type", "N/A"),
                traj_result.get("num_segments", 0),
                traj_result.get("avg_ate", 0),
                traj_result.get("avg_t_rte", 0),
                traj_result.get("avg_d_rte", 0),
                traj_result.get("avg_ATE", 0),
                traj_result.get("avg_AVE", 0),
                traj_result.get("avg_P_RMSE", 0),
                traj_result.get("avg_V_RMSE", 0),
                traj_result.get("avg_X_ATE", 0),
                traj_result.get("avg_Y_ATE", 0),
                traj_result.get("avg_Z_ATE", 0),
                traj_result.get("avg_X_AVE", 0),
                traj_result.get("avg_Y_AVE", 0),
                traj_result.get("avg_Z_AVE", 0),
                full_traj.get("ate", 0),
                full_traj.get("t_rte", 0),
                full_traj.get("d_rte", 0),
                full_traj.get("ATE", 0),
                full_traj.get("AVE", 0),
                full_traj.get("P_RMSE", 0),
                full_traj.get("V_RMSE", 0),
                full_traj.get("X_ATE", 0),
                full_traj.get("Y_ATE", 0),
                full_traj.get("Z_ATE", 0),
                full_traj.get("X_AVE", 0),
                full_traj.get("Y_AVE", 0),
                full_traj.get("Z_AVE", 0),
            ]
            writer.writerow(row)

    # 3. Segment-level Metrics CSV
    if segment_metrics_all:
        seg_file = osp.join(out_dir, "segment_metrics.csv")
        with open(seg_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Segment_ID",
                    "Trajectory_ID",
                    "Data_Name",
                    "Segment_Length",
                    "Start_Index",
                    "End_Index",
                    "Num_Points",
                    "ATE",
                    "T_RTE",
                    "D_RTE",
                    "ATE_metric",
                    "AVE",
                    "P_RMSE",
                    "V_RMSE",
                    "X_ATE",
                    "Y_ATE",
                    "Z_ATE",
                    "X_AVE",
                    "Y_AVE",
                    "Z_AVE",
                ]
            )

            for i, seg_metric in enumerate(segment_metrics_all):
                row = [
                    f"Seg_{i+1:03d}",
                    f"Traj_{seg_metric.get('trajectory_id', 'N/A')}",
                    seg_metric.get("data_name", "N/A"),
                    seg_metric.get("segment_length", 0),
                    seg_metric.get("start_idx", 0),
                    seg_metric.get("end_idx", 0),
                    seg_metric.get("end_idx", 0) - seg_metric.get("start_idx", 0) + 1,
                    seg_metric.get("ate", 0),
                    seg_metric.get("t_rte", 0),
                    seg_metric.get("d_rte", 0),
                    seg_metric.get("ATE", 0),
                    seg_metric.get("AVE", 0),
                    seg_metric.get("P_RMSE", 0),
                    seg_metric.get("V_RMSE", 0),
                    seg_metric.get("X_ATE", 0),
                    seg_metric.get("Y_ATE", 0),
                    seg_metric.get("Z_ATE", 0),
                    seg_metric.get("X_AVE", 0),
                    seg_metric.get("Y_AVE", 0),
                    seg_metric.get("Z_AVE", 0),
                ]
                writer.writerow(row)

    # 4. Statistical Summary CSV
    stats_file = osp.join(out_dir, "statistical_summary.csv")
    with open(stats_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Metric_Category",
                "Metric_Name",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Median",
                "Count",
            ]
        )

        if segment_metrics_all:
            # Calculate statistics for all metrics
            metrics_to_analyze = [
                "ate",
                "P_RMSE",
                "V_RMSE",
                "t_rte",
                "d_rte",
                "ATE",
                "AVE",
                "X_ATE",
                "Y_ATE",
                "Z_ATE",
                "X_AVE",
                "Y_AVE",
                "Z_AVE",
            ]

            for metric in metrics_to_analyze:
                values = [
                    seg.get(metric, 0) for seg in segment_metrics_all if metric in seg
                ]
                if values:
                    writer.writerow(
                        [
                            "Segment_Level",
                            metric.upper(),
                            f"{np.mean(values):.6f}",
                            f"{np.std(values):.6f}",
                            f"{np.min(values):.6f}",
                            f"{np.max(values):.6f}",
                            f"{np.median(values):.6f}",
                            len(values),
                        ]
                    )

    # 5. Performance Comparison CSV
    comp_file = osp.join(out_dir, "performance_comparison.csv")
    with open(comp_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Comparison_Type",
                "Metric",
                "Segment_Average",
                "Full_Trajectory_Average",
                "Difference",
                "Percentage_Difference",
            ]
        )

        # Compare segment averages vs full trajectory averages
        seg_avg_ate = (
            np.mean([seg.get("ate", 0) for seg in segment_metrics_all])
            if segment_metrics_all
            else 0
        )
        full_avg_ate = np.mean(
            [
                traj.get("full_trajectory", {}).get("ate", 0)
                for traj in all_trajectory_results
            ]
        )

        diff_ate = full_avg_ate - seg_avg_ate
        pct_diff_ate = (diff_ate / seg_avg_ate * 100) if seg_avg_ate != 0 else 0

        writer.writerow(
            [
                "ATE_Comparison",
                "ATE",
                f"{seg_avg_ate:.6f}",
                f"{full_avg_ate:.6f}",
                f"{diff_ate:.6f}",
                f"{pct_diff_ate:.2f}%",
            ]
        )

        # Add more comparisons as needed
        seg_avg_rmse = (
            np.mean([seg.get("P_RMSE", 0) for seg in segment_metrics_all])
            if segment_metrics_all
            else 0
        )
        full_avg_rmse = np.mean(
            [
                traj.get("full_trajectory", {}).get("P_RMSE", 0)
                for traj in all_trajectory_results
            ]
        )

        diff_rmse = full_avg_rmse - seg_avg_rmse
        pct_diff_rmse = (diff_rmse / seg_avg_rmse * 100) if seg_avg_rmse != 0 else 0

        writer.writerow(
            [
                "RMSE_Comparison",
                "P_RMSE",
                f"{seg_avg_rmse:.6f}",
                f"{full_avg_rmse:.6f}",
                f"{diff_rmse:.6f}",
                f"{pct_diff_rmse:.2f}%",
            ]
        )

    success(f"✅ Comprehensive CSV files saved to {out_dir}:")
    info(f"   📊 overall_summary.csv - Overall test statistics")
    info(f"   🚀 trajectory_metrics.csv - Individual trajectory results")
    info(f"   📏 segment_metrics.csv - Segment-level detailed metrics")
    info(f"   📈 statistical_summary.csv - Statistical analysis of all metrics")
    info(f"   🔍 performance_comparison.csv - Segment vs Full trajectory comparison")


def create_segments_summary_plot(all_trajectory_results, out_dir):
    """
    Create a summary plot showing all trajectory segments.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 12))

    # Plot 1: Segment ATE distribution
    plt.subplot(2, 3, 1)
    all_ate = []
    segment_labels = []
    for i, traj_result in enumerate(all_trajectory_results):
        for j, seg_metric in enumerate(traj_result["segment_metrics"]):
            all_ate.append(seg_metric["ate"])
            segment_labels.append(f"T{i}_S{j}")

    plt.hist(all_ate, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("ATE (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segment ATE")
    plt.grid(True, alpha=0.3)

    # Plot 2: Segment RMSE distribution
    plt.subplot(2, 3, 2)
    all_rmse = [
        seg_metric["P_RMSE"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]
    plt.hist(all_rmse, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
    plt.xlabel("P_RMSE (m)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segment P_RMSE")
    plt.grid(True, alpha=0.3)

    # Plot 3: Segment length vs ATE
    plt.subplot(2, 3, 3)
    segment_lengths = [
        seg_metric["segment_length"]
        for traj_result in all_trajectory_results
        for seg_metric in traj_result["segment_metrics"]
    ]
    plt.scatter(segment_lengths, all_ate, alpha=0.6, color="green")
    plt.xlabel("Segment Length (m)")
    plt.ylabel("ATE (m)")
    plt.title("Segment Length vs ATE")
    plt.grid(True, alpha=0.3)

    # Plot 4: Trajectory-wise average ATE
    plt.subplot(2, 3, 4)
    traj_avg_ate = [
        np.mean([seg["ate"] for seg in traj["segment_metrics"]])
        for traj in all_trajectory_results
    ]
    traj_names = [f"Traj_{i}" for i in range(len(all_trajectory_results))]
    plt.bar(traj_names, traj_avg_ate, color="orange", alpha=0.7)
    plt.xlabel("Trajectory")
    plt.ylabel("Average ATE (m)")
    plt.title("Average ATE per Trajectory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 5: Metrics comparison
    plt.subplot(2, 3, 5)
    metrics = ["ate", "t_rte", "d_rte", "P_RMSE", "V_RMSE"]
    metric_names = ["ATE", "T_RTE", "D_RTE", "P_RMSE", "V_RMSE"]
    avg_metrics = []

    for metric in metrics:
        values = [
            seg_metric[metric]
            for traj_result in all_trajectory_results
            for seg_metric in traj_result["segment_metrics"]
        ]
        avg_metrics.append(np.mean(values))

    plt.bar(
        metric_names,
        avg_metrics,
        color=["red", "blue", "green", "purple", "orange"],
        alpha=0.7,
    )
    plt.ylabel("Average Value")
    plt.title("Average Metrics Across All Segments")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 6: Segment count per trajectory
    plt.subplot(2, 3, 6)
    segment_counts = [len(traj["segment_metrics"]) for traj in all_trajectory_results]
    plt.bar(traj_names, segment_counts, color="lightblue", alpha=0.7)
    plt.xlabel("Trajectory")
    plt.ylabel("Number of Segments")
    plt.title("Segment Count per Trajectory")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "segments_summary_plot.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create additional summary statistics
    create_segments_summary_statistics(all_trajectory_results, out_dir)

    # Create 3D summary plot showing all segments aligned to origin
    create_3d_segments_summary(all_trajectory_results, out_dir)


def create_segments_summary_statistics(all_trajectory_results, out_dir):
    """
    Create summary statistics for all segments.

    Args:
        all_trajectory_results: List of all trajectory results
        out_dir: Output directory
    """
    import matplotlib.pyplot as plt

    # Collect all segment metrics
    all_segments = []
    for traj_result in all_trajectory_results:
        all_segments.extend(traj_result["segment_metrics"])

    if not all_segments:
        return

    # Create statistics summary
    metrics_to_analyze = ["ate", "t_rte", "d_rte", "P_RMSE", "V_RMSE", "segment_length"]
    metric_names = ["ATE", "T_RTE", "D_RTE", "P_RMSE", "V_RMSE", "Segment Length"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics_to_analyze, metric_names)):
        values = [seg[metric] for seg in all_segments]

        # Box plot
        axes[i].boxplot(
            values, patch_artist=True, boxprops=dict(facecolor="lightblue", alpha=0.7)
        )
        axes[i].set_title(f"{name} Distribution")
        axes[i].set_ylabel(name)
        axes[i].grid(True, alpha=0.3)

        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)

        stats_text = f"Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}"
        axes[i].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(
        osp.join(out_dir, "segments_statistics.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


class tester(object):
    def __init__(self, args, cfg, model):
        super(tester, self).__init__()
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.out_dir = cfg["test"]["out_dir"]  # +'/'+args.exp_name

        self.pred_velocity = cfg["model"]["pred_velocity"]
        self.window_time = cfg["model_param"]["window_time"]
        self.start_cov_epochs = cfg["train"]["start_cov_epochs"]
        self.plot_cnt = 0
        self._processed_trajectories = (
            []
        )  # Track processed trajectories for ID assignment

    def inference_step(self, data_loader, epoch, resume_model=None):
        (
            targets_all,
            preds_all,
            preds_cov_all,
            preds_model_cov_all,
            losses_all,
            mse_recovers,
            pred_aug_error_all,
            orien_all,
        ) = ([], [], [], [], [], [], [], [])
        for bid, batch in tqdm(enumerate(data_loader)):
            batch = [t.to(self.device) for t in batch]
            torch.cuda.synchronize()
            data_end = time.time()
            pred, pred_cov, targ, ori, loss = function.fun_test_forward(
                self.cfg, resume_model, batch, self.start_cov_epochs, epoch
            )
            torch.cuda.synchronize()
            back_end = time.time()
            inference_time = back_end - data_end
            # console.log(f"inference time: {inference_time}")
            # outfile2 = osp.join("/ocean/projects/cis220039p/shiboz/shibo_code/PL-IMU/", "inference_time.txt")
            # np.savetxt(outfile2, inferback_time, delimiter=",")
            targets_all.append(torch_to_numpy(targ))
            orien_all.append(torch_to_numpy(ori))
            preds_all.append(torch_to_numpy(pred))
            preds_cov_all.append(torch_to_numpy(pred_cov))
            losses_all.append(np.mean(torch_to_numpy(loss)))
        targets_all = np.concatenate(targets_all, axis=0)
        orien_all = np.concatenate(orien_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        preds_cov_all = np.concatenate(preds_cov_all, axis=0)
        attr_dict = {
            "targets": targets_all,
            "orien": orien_all,
            "preds": preds_all,
            "preds_cov": preds_cov_all,
            "losses": losses_all,
        }

        return attr_dict

    def process_single_trajectory(
        self, data, key, epoch_num, resume_model, segment_length=5.0
    ):
        """
        Process a single trajectory with drift correction every 5 meters.
        Uses post-inference segmentation with drift correction at segment boundaries.

        Args:
            data: Data path
            key: Motion modality key
            epoch_num: Epoch number
            resume_model: Model to test
            segment_length: Length of segments in meters

        Returns:
            Dictionary containing trajectory metrics and segment metrics
        """
        logging.info(f"Processing Path: {data}, Motion Modality: {key}")

        # Create dataloader for the full trajectory
        test_loader, test_dataset = create_test_dataloader(self.cfg, data)

        # Setup output directories
        subfolder_name = key + data.split("/")[-2]
        data_name = data.split("/")[-1]
        outdir = create_output_directories(self.out_dir, subfolder_name, data_name)

        # Get ground truth data for segmentation
        gt_positions = test_dataset.gt_pos  # Ground truth positions
        gt_timestamps = test_dataset.ts  # Ground truth timestamps

        # DRIFT CORRECTION APPROACH: Post-inference segmentation with drift correction
        logging.info("=== DRIFT CORRECTION APPROACH ===")
        logging.info("Full trajectory inference + segment-based drift correction")

        # Run inference on full trajectory first
        torch.cuda.synchronize()
        data_end = time.time()
        net_attr_dict = self.inference_step(test_loader, epoch_num, resume_model)
        torch.cuda.synchronize()
        back_end = time.time()
        inferback_time = back_end - data_end

        # Generate full trajectory
        traj_attr_dict = postprocess.pose_integrate(
            self.cfg, test_dataset, net_attr_dict, self.cfg["data"]["use_local_coord"]
        )

        # SANITY CHECK: Verify full trajectory inference starts from ground truth initial position
        logging.info("=== FULL TRAJECTORY SANITY CHECK ===")
        full_gt_initial = traj_attr_dict["pos_gt"][0]
        full_pred_initial = traj_attr_dict["pos_pred"][0]
        full_initial_diff = np.linalg.norm(full_pred_initial - full_gt_initial)
        logging.info(f"Full trajectory GT initial: {full_gt_initial}")
        logging.info(f"Full trajectory Pred initial: {full_pred_initial}")
        logging.info(f"Full trajectory initial diff: {full_initial_diff:.8f}")
        logging.info(f"Full trajectory alignment correct: {full_initial_diff <= 1e-6}")
        logging.info("=" * 50)

        # Segment trajectory after inference
        trajectory_segments = segment_trajectory_5m(traj_attr_dict, segment_length)
        logging.info(
            f"Trajectory divided into {len(trajectory_segments)} segments of ~{segment_length}m each"
        )

        # Apply drift correction to each segment
        corrected_segments = []
        for seg_idx, segment in enumerate(trajectory_segments):
            logging.info(f"Applying drift correction to segment {seg_idx}")

            # Get ground truth initial position for this segment
            gt_initial_pos = segment["pos_gt"][0]
            pred_initial_pos = segment["pos_pred"][0]

            # Apply drift correction: align to ground truth initial position
            pos_pred_corrected = segment["pos_pred"] - pred_initial_pos + gt_initial_pos

            # Create corrected segment
            corrected_segment = segment.copy()
            corrected_segment["pos_pred"] = pos_pred_corrected

            # SANITY CHECK: Verify drift correction
            seg_initial_diff = np.linalg.norm(pos_pred_corrected[0] - gt_initial_pos)
            logging.info(f"Segment {seg_idx} GT anchor: {gt_initial_pos}")
            logging.info(
                f"Segment {seg_idx} Pred start (corrected): {pos_pred_corrected[0]}"
            )
            logging.info(
                f"Segment {seg_idx} drift correction: {'✅ SUCCESS' if seg_initial_diff <= 1e-6 else '❌ FAILED'}"
            )

            corrected_segments.append(corrected_segment)

        # FIX: Apply drift correction to the FULL trajectory for visualization
        logging.info("=== APPLYING DRIFT CORRECTION TO FULL TRAJECTORY ===")
        full_gt_initial = traj_attr_dict["pos_gt"][0]
        full_pred_initial = traj_attr_dict["pos_pred"][0]

        # Apply drift correction to full trajectory: align to ground truth initial position
        traj_attr_dict_corrected = traj_attr_dict.copy()
        traj_attr_dict_corrected["pos_pred"] = (
            traj_attr_dict["pos_pred"] - full_pred_initial + full_gt_initial
        )

        # SANITY CHECK: Verify full trajectory drift correction
        full_corrected_initial = traj_attr_dict_corrected["pos_pred"][0]
        full_corrected_diff = np.linalg.norm(full_corrected_initial - full_gt_initial)
        logging.info(f"Full trajectory GT initial: {full_gt_initial}")
        logging.info(
            f"Full trajectory Pred initial (corrected): {full_corrected_initial}"
        )
        logging.info(
            f"Full trajectory drift correction: {'✅ SUCCESS' if full_corrected_diff <= 1e-6 else '❌ FAILED'}"
        )
        logging.info("=" * 50)

        # Process corrected segments with trajectory information
        trajectory_info = {
            "data_name": data_name,
            "robot_type": key,
            "trajectory_id": len(self._processed_trajectories) + 1,
        }
        segment_metrics = process_trajectory_segments(
            corrected_segments, outdir, self.cfg, trajectory_info
        )

        # Compute aggregated metrics
        aggregated_metrics = compute_aggregated_metrics(segment_metrics)

        # Save full trajectory data (using corrected trajectory)
        plot_dict = postprocess.compute_plot_dict(
            self.cfg["data"]["sample_freq"], net_attr_dict, traj_attr_dict_corrected
        )
        save_full_trajectory_data(
            traj_attr_dict_corrected, outdir, epoch_num, plot_dict
        )

        # Compute full trajectory metrics (using corrected trajectory)
        full_metrics = compute_full_trajectory_metrics(
            traj_attr_dict_corrected, self.cfg
        )

        # Prepare results
        trajectory_results = {
            "data": data_name,
            "num_segments": len(segment_metrics),
            "robot_type": key,
            "processing_path": data,
            "processing_method": "post_inference_with_drift_correction",
            "segment_metrics": segment_metrics,
            "full_trajectory": full_metrics,
            "output_dir": outdir,
            **aggregated_metrics,
        }

        # Save metrics
        save_metrics_files(trajectory_results, outdir, data_name, segment_metrics)

        # Generate plots
        # Pass the computed metrics instead of empty lists
        ave_ate_list = [full_metrics["ate"]] if "ate" in full_metrics else []
        t_rte_list = [full_metrics["t_rte"]] if "t_rte" in full_metrics else []
        d_rte_list = [full_metrics["d_rte"]] if "d_rte" in full_metrics else []

        postprocess.make_plots(
            plot_dict,
            outdir,
            epoch_num,
            ave_ate_list,
            t_rte_list,
            d_rte_list,
            use_local=self.cfg["data"]["use_local_coord"],
        )

        return trajectory_results, aggregated_metrics

    def test(
        self,
        test_data_path_list,
        epoch_num,
        resume_model,
        ratio=None,
        segment_length=5.0,
    ):
        all_trajectory_results = []
        segment_metrics_all = []
        all_metrics = {}

        logging_config.print_columns(test_data_path_list)

        # Process each trajectory
        for key in test_data_path_list:
            for data in test_data_path_list[key]:
                try:
                    trajectory_results, aggregated_metrics = self.process_single_trajectory(
                        data, key, epoch_num, resume_model, segment_length
                    )
                except Exception as e:
                    logging.warning(f"Skipping trajectory {data}: {e}")
                    continue

                all_trajectory_results.append(trajectory_results)
                segment_metrics_all.extend(trajectory_results["segment_metrics"])
                self._processed_trajectories.append(
                    trajectory_results
                )  # Track processed trajectories

                # Log trajectory statistics
                console.log("Traj Testing Statistics:", style="bold")
                test_statistics = {
                    "num_segments": trajectory_results["num_segments"],
                    **aggregated_metrics,
                }
                logging_config.console_log(test_statistics)

                # Append to main metrics file
                with open(self.out_dir + "/metrics.json", "a") as f:
                    json.dump({"cur_traj_data": trajectory_results}, f, indent=1)

        # Compute overall statistics
        overall_stats = compute_overall_statistics(
            all_trajectory_results, segment_metrics_all
        )
        all_metrics["all_traj"] = overall_stats

        # Display comprehensive rich tables
        from utils.rich_logging import banner

        banner("📊 COMPREHENSIVE TEST RESULTS")
        display_rich_metrics_tables(
            all_trajectory_results, overall_stats, segment_metrics_all
        )

        # Save comprehensive CSV files
        save_comprehensive_csv_files(
            all_trajectory_results, overall_stats, segment_metrics_all, self.out_dir
        )

        # # Save overall segment metrics (JSON format for compatibility)
        # with open(self.out_dir + "/all_segment_metrics.json", "w") as f:
        #     json.dump(segment_metrics_all, f, indent=1)

        # Create summary plot of all segments
        create_segments_summary_plot(all_trajectory_results, self.out_dir)

        # Log results to wandb if available
        self.log_wandb_test_results(all_trajectory_results, overall_stats, segment_metrics_all)

        return all_metrics

    def log_wandb_test_results(self, all_trajectory_results, overall_stats, segment_metrics_all):
        """Log ATE/RTE tables and summary plots to wandb after testing."""
        if wandb.run is None:
            return

        # 1. Log overall summary metrics
        summary = {}
        for key in ["avg_ate", "avg_t_rte", "avg_d_rte", "avg_ATE", "avg_AVE", "avg_P_RMSE", "avg_V_RMSE"]:
            if key in overall_stats:
                summary[f"test/{key}"] = overall_stats[key]
        if "segment_statistics" in overall_stats:
            seg_stats = overall_stats["segment_statistics"]
            summary["test/total_segments"] = seg_stats["total_segments"]
            summary["test/avg_segment_length"] = seg_stats["avg_segment_length"]
            summary["test/segment_ate_mean"] = seg_stats["segment_ate_stats"]["mean"]
            summary["test/segment_ate_std"] = seg_stats["segment_ate_stats"]["std"]
            summary["test/segment_rmse_mean"] = seg_stats["segment_rmse_stats"]["mean"]
        wandb.log(summary)

        # 2. Log per-trajectory ATE/RTE table
        traj_columns = ["trajectory", "data", "segments", "avg_ate", "avg_t_rte", "avg_d_rte", "avg_P_RMSE", "avg_V_RMSE"]
        traj_data = []
        for i, traj in enumerate(all_trajectory_results):
            traj_data.append([
                f"Traj_{i+1}",
                traj.get("data", "N/A"),
                traj.get("num_segments", 0),
                round(traj.get("avg_ate", 0), 4),
                round(traj.get("avg_t_rte", 0), 4),
                round(traj.get("avg_d_rte", 0), 4),
                round(traj.get("avg_P_RMSE", 0), 4),
                round(traj.get("avg_V_RMSE", 0), 4),
            ])
        wandb.log({"test/trajectory_metrics": wandb.Table(columns=traj_columns, data=traj_data)})

        # 3. Log per-segment ATE/RTE table
        if segment_metrics_all:
            seg_columns = ["segment", "trajectory", "segment_length", "ate", "t_rte", "d_rte", "P_RMSE", "V_RMSE"]
            seg_data = []
            for i, seg in enumerate(segment_metrics_all):
                seg_data.append([
                    i,
                    seg.get("trajectory", "N/A"),
                    round(seg.get("segment_length", 0), 2),
                    round(seg.get("ate", 0), 4),
                    round(seg.get("t_rte", 0), 4),
                    round(seg.get("d_rte", 0), 4),
                    round(seg.get("P_RMSE", 0), 4),
                    round(seg.get("V_RMSE", 0), 4),
                ])
            wandb.log({"test/segment_metrics": wandb.Table(columns=seg_columns, data=seg_data)})

        # 4. Log summary plots as images
        plot_files = [
            osp.join(self.out_dir, "segments_summary_plot.png"),
            osp.join(self.out_dir, "segments_statistics.png"),
        ]
        for plot_path in plot_files:
            if osp.exists(plot_path):
                wandb.log({f"test/{osp.basename(plot_path)}": wandb.Image(plot_path)})

        # 5. Log per-trajectory 3D plots and segment plots
        for i, traj in enumerate(all_trajectory_results):
            traj_dir = traj.get("output_dir", "")
            if traj_dir and osp.isdir(traj_dir):
                for fname in os.listdir(traj_dir):
                    if fname.endswith(".png"):
                        img_path = osp.join(traj_dir, fname)
                        wandb.log({f"test/traj_{i+1}/{fname}": wandb.Image(img_path)})

        logging.info("Test results logged to wandb")
