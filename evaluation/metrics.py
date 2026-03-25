"""
Evaluation metrics for neural inertial tracking.

This module provides metrics for evaluating trajectory estimation performance,
including ATE, RPE, orientation errors, and other standard SLAM metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    """Container for trajectory evaluation metrics."""

    # Absolute Trajectory Error (ATE)
    ate_mean: float
    ate_std: float
    ate_median: float
    ate_max: float
    ate_min: float

    # Relative Pose Error (RPE) - translation
    rpe_trans_mean: float
    rpe_trans_std: float
    rpe_trans_median: float

    # Relative Pose Error (RPE) - rotation
    rpe_rot_mean: float
    rpe_rot_std: float
    rpe_rot_median: float

    # Additional metrics
    trajectory_length: float
    num_poses: int
    completion_ratio: float


@dataclass
class PoseMetrics:
    """Container for pose estimation metrics."""

    # Translation errors
    trans_error_mean: float
    trans_error_std: float
    trans_error_max: float

    # Rotation errors (degrees)
    rot_error_mean: float
    rot_error_std: float
    rot_error_max: float

    # Per-axis translation errors
    trans_x_error: float
    trans_y_error: float
    trans_z_error: float

    # Per-axis rotation errors
    rot_x_error: float
    rot_y_error: float
    rot_z_error: float


@dataclass
class SequenceMetrics:
    """Container for sequence-level evaluation metrics."""

    # Trajectory metrics
    trajectory: TrajectoryMetrics

    # Pose metrics
    pose: PoseMetrics

    # Sequence information
    sequence_name: str
    sequence_length: int
    total_time: float

    # Computational metrics
    inference_time: Optional[float] = None
    fps: Optional[float] = None


def compute_ate(
    gt_poses: np.ndarray, pred_poses: np.ndarray, align_trajectories: bool = True
) -> Tuple[float, float, float, float, float]:
    """
    Compute Absolute Trajectory Error (ATE).

    Args:
        gt_poses: Ground truth poses [N, 7] (pos_xyz, quat_wxyz)
        pred_poses: Predicted poses [N, 7] (pos_xyz, quat_wxyz)
        align_trajectories: Whether to align trajectories using Umeyama algorithm

    Returns:
        Tuple of (mean, std, median, max, min) ATE values
    """
    if gt_poses.shape != pred_poses.shape:
        raise ValueError(
            f"Pose arrays must have same shape: {gt_poses.shape} vs {pred_poses.shape}"
        )

    if gt_poses.shape[1] != 7:
        raise ValueError(
            f"Poses must have 7 columns (pos_xyz, quat_wxyz), got {gt_poses.shape[1]}"
        )

    gt_positions = gt_poses[:, :3]
    pred_positions = pred_poses[:, :3]

    # Align trajectories if requested
    if align_trajectories:
        pred_positions = _align_trajectories_umeyama(gt_positions, pred_positions)

    # Compute translation errors
    translation_errors = np.linalg.norm(gt_positions - pred_positions, axis=1)

    ate_mean = np.mean(translation_errors)
    ate_std = np.std(translation_errors)
    ate_median = np.median(translation_errors)
    ate_max = np.max(translation_errors)
    ate_min = np.min(translation_errors)

    return ate_mean, ate_std, ate_median, ate_max, ate_min


def compute_rpe(
    gt_poses: np.ndarray, pred_poses: np.ndarray, delta: int = 1
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute Relative Pose Error (RPE).

    Args:
        gt_poses: Ground truth poses [N, 7] (pos_xyz, quat_wxyz)
        pred_poses: Predicted poses [N, 7] (pos_xyz, quat_wxyz)
        delta: Step size for relative pose computation

    Returns:
        Tuple of ((trans_mean, trans_std, trans_median), (rot_mean, rot_std, rot_median))
    """
    if len(gt_poses) < delta + 1:
        raise ValueError(
            f"Not enough poses for delta={delta}, need at least {delta + 1}"
        )

    translation_errors = []
    rotation_errors = []

    for i in range(len(gt_poses) - delta):
        # Ground truth relative pose
        gt_rel = _compute_relative_pose(gt_poses[i], gt_poses[i + delta])

        # Predicted relative pose
        pred_rel = _compute_relative_pose(pred_poses[i], pred_poses[i + delta])

        # Relative error
        rel_error = _compute_relative_pose_error(gt_rel, pred_rel)

        translation_errors.append(np.linalg.norm(rel_error[:3]))
        rotation_errors.append(np.abs(rel_error[6]))  # Quaternion w component for angle

    translation_errors = np.array(translation_errors)
    rotation_errors = np.array(rotation_errors) * 2  # Convert to angle
    rotation_errors = np.rad2deg(np.arccos(np.clip(rotation_errors, 0, 1)))

    trans_stats = (
        np.mean(translation_errors),
        np.std(translation_errors),
        np.median(translation_errors),
    )
    rot_stats = (
        np.mean(rotation_errors),
        np.std(rotation_errors),
        np.median(rotation_errors),
    )

    return trans_stats, rot_stats


def compute_orientation_error(
    gt_poses: np.ndarray, pred_poses: np.ndarray
) -> np.ndarray:
    """
    Compute orientation errors between ground truth and predicted poses.

    Args:
        gt_poses: Ground truth poses [N, 7] (pos_xyz, quat_wxyz)
        pred_poses: Predicted poses [N, 7] (pos_xyz, quat_wxyz)

    Returns:
        Orientation errors in degrees [N]
    """
    gt_quats = gt_poses[:, 3:7]  # quat_wxyz
    pred_quats = pred_poses[:, 3:7]

    # Normalize quaternions
    gt_quats = gt_quats / np.linalg.norm(gt_quats, axis=1, keepdims=True)
    pred_quats = pred_quats / np.linalg.norm(pred_quats, axis=1, keepdims=True)

    # Compute relative rotations
    errors = []
    for i in range(len(gt_quats)):
        # Quaternion difference
        q_diff = _quaternion_multiply(_quaternion_conjugate(gt_quats[i]), pred_quats[i])

        # Convert to angle
        angle = 2 * np.arccos(np.abs(np.clip(q_diff[0], -1, 1)))  # w component
        errors.append(np.rad2deg(angle))

    return np.array(errors)


def compute_translation_error(
    gt_poses: np.ndarray, pred_poses: np.ndarray
) -> np.ndarray:
    """
    Compute translation errors between ground truth and predicted poses.

    Args:
        gt_poses: Ground truth poses [N, 7] (pos_xyz, quat_wxyz)
        pred_poses: Predicted poses [N, 7] (pos_xyz, quat_wxyz)

    Returns:
        Translation errors [N]
    """
    gt_positions = gt_poses[:, :3]
    pred_positions = pred_poses[:, :3]

    return np.linalg.norm(gt_positions - pred_positions, axis=1)


def compute_position_only_trajectory_metrics(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    align_trajectories: bool = True,
) -> TrajectoryMetrics:
    """
    Compute trajectory metrics for position-only data (no orientation).

    Args:
        gt_positions: Ground truth positions [N, 3]
        pred_positions: Predicted positions [N, 3]
        align_trajectories: Whether to align trajectories

    Returns:
        TrajectoryMetrics object
    """
    if align_trajectories:
        pred_positions = _align_trajectories_umeyama(gt_positions, pred_positions)

    # Compute ATE
    position_errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
    ate_mean = np.mean(position_errors)
    ate_std = np.std(position_errors)
    ate_median = np.median(position_errors)
    ate_max = np.max(position_errors)
    ate_min = np.min(position_errors)

    # Compute RPE (simplified for position-only)
    if len(gt_positions) > 1:
        gt_diffs = np.diff(gt_positions, axis=0)
        pred_diffs = np.diff(pred_positions, axis=0)

        gt_diffs_norm = np.linalg.norm(gt_diffs, axis=1)
        pred_diffs_norm = np.linalg.norm(pred_diffs, axis=1)

        rpe_trans_errors = np.abs(gt_diffs_norm - pred_diffs_norm)
        rpe_trans_mean = np.mean(rpe_trans_errors)
        rpe_trans_std = np.std(rpe_trans_errors)
        rpe_trans_median = np.median(rpe_trans_errors)
    else:
        rpe_trans_mean = rpe_trans_std = rpe_trans_median = 0.0

    # Placeholder rotation metrics (not applicable for position-only)
    rpe_rot_mean = rpe_rot_std = rpe_rot_median = 0.0

    # Trajectory length
    trajectory_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))
    num_poses = len(gt_positions)
    completion_ratio = 1.0  # Assuming full completion

    return TrajectoryMetrics(
        ate_mean=ate_mean,
        ate_std=ate_std,
        ate_median=ate_median,
        ate_max=ate_max,
        ate_min=ate_min,
        rpe_trans_mean=rpe_trans_mean,
        rpe_trans_std=rpe_trans_std,
        rpe_trans_median=rpe_trans_median,
        rpe_rot_mean=rpe_rot_mean,
        rpe_rot_std=rpe_rot_std,
        rpe_rot_median=rpe_rot_median,
        trajectory_length=trajectory_length,
        num_poses=num_poses,
        completion_ratio=completion_ratio,
    )


def compute_position_and_orientation_pose_metrics(
    gt_positions: np.ndarray,
    pred_positions: np.ndarray,
    gt_orientations: np.ndarray,
    pred_orientations: np.ndarray,
) -> PoseMetrics:
    """
    Compute pose metrics for position and orientation data.

    Args:
        gt_positions: Ground truth positions [N, 3]
        pred_positions: Predicted positions [N, 3]
        gt_orientations: Ground truth orientations [N, 3] (Euler angles in radians)
        pred_orientations: Predicted orientations [N, 3] (Euler angles in radians)

    Returns:
        PoseMetrics object
    """
    # Translation errors
    trans_errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
    trans_error_mean = np.mean(trans_errors)
    trans_error_std = np.std(trans_errors)
    trans_error_max = np.max(trans_errors)

    # Per-axis translation errors
    axis_errors = gt_positions - pred_positions
    trans_x_error = np.mean(np.abs(axis_errors[:, 0]))
    trans_y_error = np.mean(np.abs(axis_errors[:, 1]))
    trans_z_error = np.mean(np.abs(axis_errors[:, 2]))

    # Rotation errors (convert to degrees for easier interpretation)
    # Compute angular differences for each axis
    rot_errors_rad = np.abs(gt_orientations - pred_orientations)
    # Normalize angles to [-π, π] range
    rot_errors_rad = np.mod(rot_errors_rad + np.pi, 2 * np.pi) - np.pi
    rot_errors_deg = rot_errors_rad * 180 / np.pi

    rot_error_mean = np.mean(rot_errors_deg)
    rot_error_std = np.std(rot_errors_deg)
    rot_error_max = np.max(rot_errors_deg)

    # Per-axis rotation errors (in degrees)
    rot_x_error = np.mean(rot_errors_deg[:, 0])
    rot_y_error = np.mean(rot_errors_deg[:, 1])
    rot_z_error = np.mean(rot_errors_deg[:, 2])

    return PoseMetrics(
        trans_error_mean=trans_error_mean,
        trans_error_std=trans_error_std,
        trans_error_max=trans_error_max,
        rot_error_mean=rot_error_mean,
        rot_error_std=rot_error_std,
        rot_error_max=rot_error_max,
        trans_x_error=trans_x_error,
        trans_y_error=trans_y_error,
        trans_z_error=trans_z_error,
        rot_x_error=rot_x_error,
        rot_y_error=rot_y_error,
        rot_z_error=rot_z_error,
    )


def compute_position_only_pose_metrics(
    gt_positions: np.ndarray, pred_positions: np.ndarray
) -> PoseMetrics:
    """
    Compute pose metrics for position-only data (no orientation).

    Args:
        gt_positions: Ground truth positions [N, 3]
        pred_positions: Predicted positions [N, 3]

    Returns:
        PoseMetrics object
    """
    # Translation errors
    trans_errors = np.linalg.norm(gt_positions - pred_positions, axis=1)
    trans_error_mean = np.mean(trans_errors)
    trans_error_std = np.std(trans_errors)
    trans_error_max = np.max(trans_errors)

    # Per-axis translation errors
    axis_errors = gt_positions - pred_positions
    trans_x_error = np.mean(np.abs(axis_errors[:, 0]))
    trans_y_error = np.mean(np.abs(axis_errors[:, 1]))
    trans_z_error = np.mean(np.abs(axis_errors[:, 2]))

    # Placeholder rotation errors (not applicable for position-only)
    rot_error_mean = rot_error_std = rot_error_max = 0.0
    rot_x_error = rot_y_error = rot_z_error = 0.0

    return PoseMetrics(
        trans_error_mean=trans_error_mean,
        trans_error_std=trans_error_std,
        trans_error_max=trans_error_max,
        rot_error_mean=rot_error_mean,
        rot_error_std=rot_error_std,
        rot_error_max=rot_error_max,
        trans_x_error=trans_x_error,
        trans_y_error=trans_y_error,
        trans_z_error=trans_z_error,
        rot_x_error=rot_x_error,
        rot_y_error=rot_y_error,
        rot_z_error=rot_z_error,
    )


def compute_trajectory_metrics(
    gt_poses: np.ndarray, pred_poses: np.ndarray, align_trajectories: bool = True
) -> TrajectoryMetrics:
    """
    Compute comprehensive trajectory evaluation metrics.

    Args:
        gt_poses: Ground truth poses [N, 7]
        pred_poses: Predicted poses [N, 7]
        align_trajectories: Whether to align trajectories

    Returns:
        TrajectoryMetrics object
    """
    # ATE computation
    ate_mean, ate_std, ate_median, ate_max, ate_min = compute_ate(
        gt_poses, pred_poses, align_trajectories
    )

    # RPE computation
    (rpe_trans_mean, rpe_trans_std, rpe_trans_median), (
        rpe_rot_mean,
        rpe_rot_std,
        rpe_rot_median,
    ) = compute_rpe(gt_poses, pred_poses)

    # Trajectory length
    gt_positions = gt_poses[:, :3]
    trajectory_length = np.sum(np.linalg.norm(np.diff(gt_positions, axis=0), axis=1))

    # Completion ratio (assume full trajectory for now)
    completion_ratio = 1.0

    return TrajectoryMetrics(
        ate_mean=ate_mean,
        ate_std=ate_std,
        ate_median=ate_median,
        ate_max=ate_max,
        ate_min=ate_min,
        rpe_trans_mean=rpe_trans_mean,
        rpe_trans_std=rpe_trans_std,
        rpe_trans_median=rpe_trans_median,
        rpe_rot_mean=rpe_rot_mean,
        rpe_rot_std=rpe_rot_std,
        rpe_rot_median=rpe_rot_median,
        trajectory_length=trajectory_length,
        num_poses=len(gt_poses),
        completion_ratio=completion_ratio,
    )


def compute_pose_metrics(gt_poses: np.ndarray, pred_poses: np.ndarray) -> PoseMetrics:
    """
    Compute pose estimation metrics.

    Args:
        gt_poses: Ground truth poses [N, 7]
        pred_poses: Predicted poses [N, 7]

    Returns:
        PoseMetrics object
    """
    # Translation errors
    trans_errors = compute_translation_error(gt_poses, pred_poses)
    trans_error_mean = np.mean(trans_errors)
    trans_error_std = np.std(trans_errors)
    trans_error_max = np.max(trans_errors)

    # Per-axis translation errors
    gt_positions = gt_poses[:, :3]
    pred_positions = pred_poses[:, :3]
    trans_diff = gt_positions - pred_positions

    trans_x_error = np.mean(np.abs(trans_diff[:, 0]))
    trans_y_error = np.mean(np.abs(trans_diff[:, 1]))
    trans_z_error = np.mean(np.abs(trans_diff[:, 2]))

    # Rotation errors
    rot_errors = compute_orientation_error(gt_poses, pred_poses)
    rot_error_mean = np.mean(rot_errors)
    rot_error_std = np.std(rot_errors)
    rot_error_max = np.max(rot_errors)

    # Per-axis rotation errors (simplified using Euler angles)
    gt_eulers = _quaternions_to_euler(gt_poses[:, 3:7])
    pred_eulers = _quaternions_to_euler(pred_poses[:, 3:7])
    euler_diff = np.rad2deg(np.abs(gt_eulers - pred_eulers))

    rot_x_error = np.mean(euler_diff[:, 0])
    rot_y_error = np.mean(euler_diff[:, 1])
    rot_z_error = np.mean(euler_diff[:, 2])

    return PoseMetrics(
        trans_error_mean=trans_error_mean,
        trans_error_std=trans_error_std,
        trans_error_max=trans_error_max,
        rot_error_mean=rot_error_mean,
        rot_error_std=rot_error_std,
        rot_error_max=rot_error_max,
        trans_x_error=trans_x_error,
        trans_y_error=trans_y_error,
        trans_z_error=trans_z_error,
        rot_x_error=rot_x_error,
        rot_y_error=rot_y_error,
        rot_z_error=rot_z_error,
    )


# Helper functions


def _align_trajectories_umeyama(
    gt_positions: np.ndarray, pred_positions: np.ndarray
) -> np.ndarray:
    """Align trajectories using Umeyama algorithm."""
    # Simplified alignment - center and scale
    gt_centered = gt_positions - np.mean(gt_positions, axis=0)
    pred_centered = pred_positions - np.mean(pred_positions, axis=0)

    # Scale alignment
    gt_scale = np.sqrt(np.sum(gt_centered**2))
    pred_scale = np.sqrt(np.sum(pred_centered**2))

    if pred_scale > 0:
        scale = gt_scale / pred_scale
        pred_centered *= scale

    # Translation alignment
    pred_aligned = pred_centered + np.mean(gt_positions, axis=0)

    return pred_aligned


def _compute_relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
    """Compute relative pose between two poses."""
    # Extract positions and quaternions
    pos1, quat1 = pose1[:3], pose1[3:7]
    pos2, quat2 = pose2[:3], pose2[3:7]

    # Relative position
    rel_pos = pos2 - pos1

    # Relative rotation
    quat1_conj = _quaternion_conjugate(quat1)
    rel_quat = _quaternion_multiply(quat1_conj, quat2)

    return np.concatenate([rel_pos, rel_quat])


def _compute_relative_pose_error(
    gt_rel: np.ndarray, pred_rel: np.ndarray
) -> np.ndarray:
    """Compute error between relative poses."""
    # Position error
    pos_error = gt_rel[:3] - pred_rel[:3]

    # Rotation error
    gt_quat = gt_rel[3:7]
    pred_quat = pred_rel[3:7]
    quat_error = _quaternion_multiply(_quaternion_conjugate(gt_quat), pred_quat)

    return np.concatenate([pos_error, quat_error])


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def _quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quaternions_to_euler(quats: np.ndarray) -> np.ndarray:
    """Convert quaternions to Euler angles (roll, pitch, yaw)."""
    # Using scipy for robust conversion
    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])  # Convert to x,y,z,w format
    euler_angles = rotations.as_euler("xyz")
    return euler_angles


def compute_sequence_metrics(
    gt_poses: np.ndarray,
    pred_poses: np.ndarray,
    sequence_name: str,
    total_time: float,
    inference_time: Optional[float] = None,
    gt_orientations: Optional[np.ndarray] = None,
    pred_orientations: Optional[np.ndarray] = None,
) -> SequenceMetrics:
    """
    Compute comprehensive sequence evaluation metrics.

    Args:
        gt_poses: Ground truth poses [N, 7] or positions [N, 3]
        pred_poses: Predicted poses [N, 7] or positions [N, 3]
        sequence_name: Name of the sequence
        total_time: Total sequence time in seconds
        inference_time: Optional inference time in seconds

    Returns:
        SequenceMetrics object
    """
    # Handle both pose format [N, 7] and position format [N, 3]
    if gt_poses.shape[1] == 3 and pred_poses.shape[1] == 3:
        # Position-only data - create enhanced metrics with orientation if available
        trajectory_metrics = compute_position_only_trajectory_metrics(
            gt_poses, pred_poses
        )
        if gt_orientations is not None and pred_orientations is not None:
            pose_metrics = compute_position_and_orientation_pose_metrics(
                gt_poses, pred_poses, gt_orientations, pred_orientations
            )
        else:
            pose_metrics = compute_position_only_pose_metrics(gt_poses, pred_poses)
    else:
        # Full pose data with quaternions
        trajectory_metrics = compute_trajectory_metrics(gt_poses, pred_poses)
        pose_metrics = compute_pose_metrics(gt_poses, pred_poses)

    fps = None
    if inference_time is not None and inference_time > 0:
        fps = len(pred_poses) / inference_time

    return SequenceMetrics(
        trajectory=trajectory_metrics,
        pose=pose_metrics,
        sequence_name=sequence_name,
        sequence_length=len(pred_poses),
        total_time=total_time,
        inference_time=inference_time,
        fps=fps,
    )


def compute_absolute_trajectory_error(est, gt):
    """
    The Absolute Trajectory Error (ATE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: estimated trajectory
        gt: ground truth trajectory. It must have the same shape as est.

    Return:
        Absolution trajectory error, which is the Root Mean Squared Error between
        two trajectories.
    """
    # ATE = np.mean(np.linalg.norm(est - gt, axis=1)) #Norm L1
    ate = np.sqrt(np.mean(np.linalg.norm(est - gt, axis=1) ** 2))  # Norm L2
    return ate  # ATE, RMSE
    # Row vector, after subtracting prediction and ground truth at each timestamp, calculate 2-norm for xyz axes together, then take the square root of the mean of the squared norm


def compute_relative_trajectory_error_time(est, gt, delta, max_delta=-1):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    if max_delta == -1:
        max_delta = est.shape[0]  # 6015
    deltas = (
        np.array([delta]) if delta > 0 else np.arange(1, min(est.shape[0], max_delta))
    )  # array([6000])
    t_rtes = np.zeros((deltas.shape[0], 2))  # (1,2)
    for i in range(deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        # est[deltas[i]:]->est last 15 estimated values  est[: -deltas[i]]->est first 15 estimated values
        err = (
            est[deltas[i] :] + gt[: -deltas[i]] - est[: -deltas[i]] - gt[deltas[i] :]
        )  # deltas[0]:6000 err:15*3
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        t_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))  # Position difference

    # The average of RTE of all window sized is returned.
    return np.mean(t_rtes)


def compute_relative_trajectory_norm_angle_error_time(est, gt, delta):
    """
    The Relative Trajectory Error (RTE) defined in:
    A Benchmark for the evaluation of RGB-D SLAM Systems
    http://ais.informatik.uni-freiburg.de/publications/papers/sturm12iros.pdf

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.
        delta: fixed window size. If set to -1, the average of all RTE up to max_delta will be computed.
        max_delta: maximum delta. If -1 is provided, it will be set to the length of trajectories.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """
    delta = int(delta)
    re_est = est[delta:] - est[:-delta]
    re_gt = gt[delta:] - gt[:-delta]

    t_rtes = np.zeros((re_est.shape[0], 2))
    norm_est = np.linalg.norm(re_est, axis=1)
    norm_gt = np.linalg.norm(re_gt, axis=1)
    t_rtes[:, 0] = np.abs(norm_est - norm_gt)
    dot = np.array([np.dot(re_est[i], re_gt[i]) for i in range(re_est.shape[0])])
    t_rtes[:, 1] = np.arccos(dot / (norm_gt * norm_est)) * 180.0 / np.pi
    return t_rtes


def compute_relative_trajectory_error_dist(est, gt, delta=1):
    """
    Almost the same as t_rte in which the length of a window is one minute, while the length of a window in d_rte is one meter(default).

    Args:
        est: the estimated trajectory
        gt: the ground truth trajectory.

    Returns:
        Relative trajectory error. This is the mean value under different delta.
    """

    gt_delta_len = np.linalg.norm(
        gt[1:] - gt[:-1], axis=1
    )  # 6014 Calculate 2-norm of pose difference between consecutive frames in gt, which is the spatial displacement distance between consecutive frames
    end_index = np.zeros((est.shape[0], 1), dtype=int)  # 6015*1

    # calculate where the 1 meter endpoint is
    j = 0
    i = 0
    current_sum = 0.0
    while i < est.shape[0]:  # 6015
        while j < gt_delta_len.shape[0]:  # 6014
            current_sum = current_sum + gt_delta_len[j]
            if current_sum >= 1.0:  # If accumulated distance exceeds 1m
                break
            j = j + 1
        if j == gt_delta_len.shape[0]:  # If the entire trajectory accumulation doesn't reach 1m
            # done
            break
        else:  # If accumulation reaches 1m
            # reach the endpoint x_{j+1} of x_i
            end_index[i] = j + 1  # The index of the end frame is j+1
            current_sum = (
                current_sum - gt_delta_len[j]
            )  # make sure current_sum < 1.0 now subtract the pose difference of the previous frame
            current_sum = current_sum - gt_delta_len[i]  # Subtract the pose difference of the next frame
            i = i + 1

    d_rtes = np.zeros(len(end_index))  # 6015
    for i in range(
        len(end_index)
    ):  # There are many values of 0 in end_index later, meaning there are no corresponding frame indices with accumulated value of 1m
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = (
            est[end_index[i]] + gt[i] - est[i] - gt[end_index[i]]
        )  # Difference from the formula in the paper by one negative sign
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        d_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

    # The average of RTE of all window sized is returned.
    return np.mean(d_rtes)


def compute_relative_trajectory_norm_angle_error_dist(est, gt, delta=1.0):

    gt_delta_len = np.linalg.norm(gt[1:] - gt[:-1], axis=1)
    end_index = np.zeros((est.shape[0], 1), dtype=int)
    # calculate where the 1 meter endpoint is
    j = 0
    i = 0
    current_sum = 0.0
    while i < est.shape[0]:
        while j < gt_delta_len.shape[0]:
            current_sum = current_sum + gt_delta_len[j]
            if current_sum >= delta:
                break
            j = j + 1
        if j == gt_delta_len.shape[0]:
            # done
            break
        else:
            # reach the endpoint x_{j+1} of x_i
            end_index[i] = j + 1
            current_sum = (
                current_sum - gt_delta_len[j]
            )  # make sure current_sum < 1.0 now
            current_sum = current_sum - gt_delta_len[i]
            i = i + 1

    d_rtes = np.zeros((len(end_index), 2))
    for i in range(len(end_index)):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        # slided through the trajectory.
        err = est[end_index[i]] + gt[i] - est[i] - gt[end_index[i]]
        # rtes[i] = np.sqrt(np.mean(err ** 2))
        d_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))

        re_est = est[end_index[i]] - est[i]
        re_gt = gt[end_index[i]] - gt[i]
        norm_est = np.linalg.norm(re_est)
        norm_gt = np.linalg.norm(re_gt)
        norm_error = np.abs(norm_est - norm_gt)
        angle_error = (
            np.arccos(np.dot(re_est[0], re_gt[0]) / (norm_gt * norm_est + 1.0e-6))
            * 180.0
            / np.pi
        )
        d_rtes[i][0] = norm_error
        d_rtes[i][1] = angle_error

    # The average of RTE of all window sized is returned.
    return d_rtes


def compute_ate_rte(est, gt, pred_per_min=200):  # pred_per_min->60*100
    """
    A convenient function to compute ATE and RTE. For sequences shorter than pred_per_min, it computes end sequence
    drift and scales the number accordingly.
    """
    ate = compute_absolute_trajectory_error(est, gt)  # est:6015*3 gt:6015*3
    if est.shape[0] < pred_per_min:  # calcuate the rte in 1 second
        print("less than one minute!")
        ratio = pred_per_min / est.shape[0]
        t_rte = (
            compute_relative_trajectory_error_time(est, gt, delta=est.shape[0] - 1)
            * ratio
        )
    else:
        t_rte = compute_relative_trajectory_error_time(est, gt, delta=pred_per_min)

    d_rte = compute_relative_trajectory_error_dist(est, gt, delta=1)

    return ate, t_rte, d_rte


def compute_accruacy_metrics(input_dict, pred_per_min=200, use_local=False):
    gt_pos, pred_pos = input_dict["pos_gt"], input_dict["pos_pred"]  # Position of the entire trajectory length
    gt_vel, pred_vel = input_dict.get("vel_gt"), input_dict.get(
        "vel_pred"
    )  # Average velocity corresponding to every 5 frames of the entire trajectory #Since we don't predict orientation, there's no rotation metric
    assert gt_pos.shape[0] == pred_pos.shape[0]
    #pos_dist = np.linalg.norm(pred_pos[1:, :] - gt_pos[1:, :], axis=1)
    pos_dist = np.linalg.norm(pred_pos - gt_pos, axis=1) #@PREVIOUSLY BUGGY!!
    ATE = np.mean(pos_dist)
    P_RMSE = np.sqrt(np.mean(pos_dist**2))
    X_ATE = np.mean(
        np.linalg.norm(pred_pos[1:, 0][:, None] - gt_pos[1:, 0][:, None], axis=1)
    )
    Y_ATE = np.mean(
        np.linalg.norm(pred_pos[1:, 1][:, None] - gt_pos[1:, 1][:, None], axis=1)
    )
    Z_ATE = np.mean(
        np.linalg.norm(pred_pos[1:, 2][:, None] - gt_pos[1:, 2][:, None], axis=1)
    )
    if use_local and gt_vel is not None and pred_vel is not None:
        assert gt_vel.shape[0] == pred_vel.shape[0]
        #vel_dist = np.linalg.norm(pred_vel[1:, :] - gt_vel[1:, :], axis=1)
        vel_dist = np.linalg.norm(pred_vel - gt_vel, axis=1) #@PREVIOUSLY BUGGY!!
        AVE = np.mean(vel_dist)
        V_RMSE = np.sqrt(np.mean(vel_dist**2))
        X_AVE = np.mean(
            np.linalg.norm(pred_vel[1:, 0][:, None] - gt_vel[1:, 0][:, None], axis=1)
        )
        Y_AVE = np.mean(
            np.linalg.norm(pred_vel[1:, 1][:, None] - gt_vel[1:, 1][:, None], axis=1)
        )
        Z_AVE = np.mean(
            np.linalg.norm(pred_vel[1:, 2][:, None] - gt_vel[1:, 2][:, None], axis=1)
        )
    else:
        AVE = 0.0
        V_RMSE = 0.0
        X_AVE = 0.0
        Y_AVE = 0.0
        Z_AVE = 0.0

    return ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE

