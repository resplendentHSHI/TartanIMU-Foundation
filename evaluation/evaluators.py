"""
Evaluators for neural inertial tracking models.

This module provides evaluator classes for systematic evaluation of models
on various datasets and benchmarks with comprehensive metrics reporting.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .metrics import (
    PoseMetrics,
    SequenceMetrics,
    TrajectoryMetrics,
    compute_pose_metrics,
    compute_sequence_metrics,
    compute_trajectory_metrics,
)

# from ..models.base_model import BaseNeuralInertialModel  # Not available in this project
# from ..datasets.base_dataset import BaseIMUDataset  # Not available in this project
# from ..core.exceptions import EvaluationError  # Not available in this project
# from ..utils.logging_utils import MetricsLogger  # Not available in this project


# Placeholder classes for compatibility
class BaseNeuralInertialModel:
    """Placeholder base model class."""

    pass


class BaseIMUDataset:
    """Placeholder base dataset class."""

    pass


class EvaluationError(Exception):
    """Placeholder evaluation error class."""

    pass


class MetricsLogger:
    """Placeholder metrics logger class."""

    def __init__(self):
        pass


logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for model evaluators."""

    def __init__(
        self, model: BaseNeuralInertialModel, device: Optional[torch.device] = None
    ):
        """
        Initialize base evaluator.

        Args:
            model: Neural network model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        # Results storage
        self.results = {}
        self.metrics_logger = MetricsLogger()

        logger.info(f"Initialized evaluator with model on device: {self.device}")

    @abstractmethod
    def evaluate(self, dataset: BaseIMUDataset, **kwargs) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Args:
            dataset: Dataset to evaluate on
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def save_results(self, output_path: Union[str, Path], format: str = "json"):
        """
        Save evaluation results to file.

        Args:
            output_path: Path to save results
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
        elif format == "csv":
            import pandas as pd

            # Flatten results for CSV
            flattened = self._flatten_results(self.results)
            df = pd.DataFrame([flattened])
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Results saved to: {output_path}")

    def _flatten_results(
        self, results: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        flattened = {}
        for key, value in results.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_results(value, new_key))
            elif isinstance(value, (TrajectoryMetrics, PoseMetrics, SequenceMetrics)):
                flattened.update(self._flatten_results(value.__dict__, new_key))
            else:
                flattened[new_key] = value
        return flattened

    def _predict_sequence(self, sequence_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict poses for a single sequence.

        Args:
            sequence_data: Dictionary containing sequence data

        Returns:
            Predicted poses tensor
        """
        with torch.no_grad():
            imu_data = sequence_data["imu"].to(self.device)

            # Add batch dimension if needed
            if imu_data.dim() == 3:
                imu_data = imu_data.unsqueeze(0)

            # Model prediction
            predictions, _ = self.model(imu_data)

            return predictions.squeeze(0).cpu()


class TrajectoryEvaluator(BaseEvaluator):
    """Evaluator for trajectory estimation tasks."""

    def __init__(
        self, model: BaseNeuralInertialModel, device: Optional[torch.device] = None
    ):
        """Initialize trajectory evaluator."""
        super().__init__(model, device)

    def evaluate(
        self,
        dataset: BaseIMUDataset,
        batch_size: int = 1,
        save_predictions: bool = False,
        align_trajectories: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on trajectory estimation task.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            save_predictions: Whether to save predictions
            align_trajectories: Whether to align trajectories for ATE

        Returns:
            Dictionary containing trajectory evaluation results
        """
        logger.info(f"Starting trajectory evaluation on {len(dataset)} sequences")

        sequence_results = []
        all_predictions = []
        all_ground_truth = []

        total_inference_time = 0

        # Evaluate each sequence
        for i, sequence_data in enumerate(dataset):
            start_time = time.time()

            try:
                # Model prediction
                predictions = self._predict_sequence(sequence_data)

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # Ground truth
                gt_poses = sequence_data["poses"]

                # Convert predictions to pose format if needed
                pred_poses = self._convert_to_poses(predictions, gt_poses.shape)

                # Compute sequence metrics
                sequence_name = sequence_data.get("sequence_name", f"seq_{i}")
                seq_metrics = compute_sequence_metrics(
                    gt_poses.numpy(),
                    pred_poses.numpy(),
                    sequence_name,
                    len(gt_poses) * 0.01,  # Assume 100Hz
                    inference_time,
                )

                sequence_results.append(seq_metrics)

                if save_predictions:
                    all_predictions.append(pred_poses)
                    all_ground_truth.append(gt_poses)

                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(dataset)} sequences")

            except Exception as e:
                logger.error(f"Failed to evaluate sequence {i}: {e}")
                continue

        # Aggregate results
        aggregate_metrics = self._aggregate_sequence_results(sequence_results)

        # Overall statistics
        self.results = {
            "dataset_info": {
                "num_sequences": len(dataset),
                "total_samples": sum(seq.sequence_length for seq in sequence_results),
                "total_inference_time": total_inference_time,
                "avg_fps": (
                    len(dataset) / total_inference_time
                    if total_inference_time > 0
                    else 0
                ),
            },
            "aggregate_metrics": aggregate_metrics,
            "sequence_results": [seq.__dict__ for seq in sequence_results],
            "evaluation_config": {
                "batch_size": batch_size,
                "align_trajectories": align_trajectories,
                "save_predictions": save_predictions,
            },
        }

        if save_predictions:
            self.results["predictions"] = all_predictions
            self.results["ground_truth"] = all_ground_truth

        logger.info(
            f"Trajectory evaluation completed. ATE mean: {aggregate_metrics['ate_mean']:.4f}m"
        )
        return self.results

    def _convert_to_poses(
        self, predictions: torch.Tensor, target_shape: torch.Size
    ) -> torch.Tensor:
        """Convert model predictions to pose format."""
        # This assumes predictions are in [seq_len, output_dim] format
        # and need to be converted to [seq_len, 7] (pos_xyz, quat_wxyz)

        if predictions.shape == target_shape:
            return predictions

        # Handle different output formats
        if predictions.shape[-1] == 6:  # Position + orientation (6DOF)
            # Add identity quaternion for missing rotation
            seq_len = predictions.shape[0]
            poses = torch.zeros(seq_len, 7)
            poses[:, :3] = predictions[:, :3]  # Position
            poses[:, 3] = 1.0  # Quaternion w
            poses[:, 4:7] = 0.0  # Quaternion xyz
            return poses
        elif predictions.shape[-1] == 7:  # Full pose
            return predictions
        else:
            # Pad or truncate to match expected shape
            seq_len = predictions.shape[0]
            poses = torch.zeros(seq_len, 7)
            min_dim = min(predictions.shape[-1], 7)
            poses[:, :min_dim] = predictions[:, :min_dim]
            if min_dim < 7:
                poses[:, 3] = 1.0  # Default quaternion w
            return poses

    def _aggregate_sequence_results(
        self, sequence_results: List[SequenceMetrics]
    ) -> Dict[str, float]:
        """Aggregate metrics across all sequences."""
        if not sequence_results:
            return {}

        # Extract trajectory metrics
        ate_means = [seq.trajectory.ate_mean for seq in sequence_results]
        ate_stds = [seq.trajectory.ate_std for seq in sequence_results]
        rpe_trans_means = [seq.trajectory.rpe_trans_mean for seq in sequence_results]
        rpe_rot_means = [seq.trajectory.rpe_rot_mean for seq in sequence_results]

        # Extract pose metrics
        trans_errors = [seq.pose.trans_error_mean for seq in sequence_results]
        rot_errors = [seq.pose.rot_error_mean for seq in sequence_results]

        return {
            "ate_mean": np.mean(ate_means),
            "ate_std": np.std(ate_means),
            "ate_median": np.median(ate_means),
            "rpe_trans_mean": np.mean(rpe_trans_means),
            "rpe_trans_std": np.std(rpe_trans_means),
            "rpe_rot_mean": np.mean(rpe_rot_means),
            "rpe_rot_std": np.std(rpe_rot_means),
            "trans_error_mean": np.mean(trans_errors),
            "trans_error_std": np.std(trans_errors),
            "rot_error_mean": np.mean(rot_errors),
            "rot_error_std": np.std(rot_errors),
            "num_sequences_evaluated": len(sequence_results),
        }


class SequenceEvaluator(BaseEvaluator):
    """Evaluator for sequence-level tasks."""

    def evaluate(
        self,
        dataset: BaseIMUDataset,
        sequence_indices: Optional[List[int]] = None,
        compute_gradients: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on specific sequences.

        Args:
            dataset: Dataset to evaluate on
            sequence_indices: Specific sequence indices to evaluate
            compute_gradients: Whether to compute gradients for analysis

        Returns:
            Dictionary containing sequence evaluation results
        """
        if sequence_indices is None:
            sequence_indices = list(range(len(dataset)))

        logger.info(f"Evaluating {len(sequence_indices)} specific sequences")

        detailed_results = {}

        for idx in sequence_indices:
            if idx >= len(dataset):
                logger.warning(f"Sequence index {idx} out of range, skipping")
                continue

            sequence_data = dataset[idx]
            sequence_name = sequence_data.get("sequence_name", f"seq_{idx}")

            # Detailed evaluation for this sequence
            with torch.set_grad_enabled(compute_gradients):
                predictions = self._predict_sequence(sequence_data)
                gt_poses = sequence_data["poses"]

                # Compute detailed metrics
                traj_metrics = compute_trajectory_metrics(
                    gt_poses.numpy(), predictions.numpy()
                )
                pose_metrics = compute_pose_metrics(
                    gt_poses.numpy(), predictions.numpy()
                )

                seq_info = (
                    dataset.get_sequence_info(idx)
                    if hasattr(dataset, "get_sequence_info")
                    else {}
                )

                detailed_results[sequence_name] = {
                    "trajectory_metrics": traj_metrics.__dict__,
                    "pose_metrics": pose_metrics.__dict__,
                    "sequence_info": seq_info,
                    "predictions": predictions.tolist() if compute_gradients else None,
                }

        self.results = {
            "evaluation_type": "sequence_detailed",
            "num_sequences": len(sequence_indices),
            "detailed_results": detailed_results,
            "summary": self._summarize_detailed_results(detailed_results),
        }

        return self.results

    def _summarize_detailed_results(
        self, detailed_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize detailed sequence results."""
        if not detailed_results:
            return {}

        # Collect metrics across sequences
        ate_values = []
        rpe_trans_values = []
        rpe_rot_values = []

        for seq_name, result in detailed_results.items():
            traj_metrics = result["trajectory_metrics"]
            ate_values.append(traj_metrics["ate_mean"])
            rpe_trans_values.append(traj_metrics["rpe_trans_mean"])
            rpe_rot_values.append(traj_metrics["rpe_rot_mean"])

        return {
            "overall_ate_mean": np.mean(ate_values),
            "overall_ate_std": np.std(ate_values),
            "overall_rpe_trans_mean": np.mean(rpe_trans_values),
            "overall_rpe_rot_mean": np.mean(rpe_rot_values),
            "best_sequence": min(
                detailed_results.keys(),
                key=lambda x: detailed_results[x]["trajectory_metrics"]["ate_mean"],
            ),
            "worst_sequence": max(
                detailed_results.keys(),
                key=lambda x: detailed_results[x]["trajectory_metrics"]["ate_mean"],
            ),
        }


class CrossValidationEvaluator(BaseEvaluator):
    """Evaluator for cross-validation evaluation."""

    def evaluate(
        self,
        dataset: BaseIMUDataset,
        k_folds: int = 5,
        shuffle: bool = True,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation evaluation.

        Args:
            dataset: Dataset to evaluate on
            k_folds: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Starting {k_folds}-fold cross-validation")

        np.random.seed(random_seed)
        indices = np.arange(len(dataset))

        if shuffle:
            np.random.shuffle(indices)

        fold_size = len(indices) // k_folds
        fold_results = []

        for fold in range(k_folds):
            logger.info(f"Evaluating fold {fold + 1}/{k_folds}")

            # Split indices
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else len(indices)

            test_indices = indices[start_idx:end_idx]
            train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])

            # Create subset datasets
            test_data = [dataset[i] for i in test_indices]

            # Evaluate on test fold
            fold_metrics = self._evaluate_fold(test_data, fold)
            fold_results.append(fold_metrics)

        # Aggregate cross-validation results
        cv_summary = self._aggregate_cv_results(fold_results)

        self.results = {
            "evaluation_type": "cross_validation",
            "k_folds": k_folds,
            "fold_results": fold_results,
            "cv_summary": cv_summary,
            "config": {"shuffle": shuffle, "random_seed": random_seed},
        }

        logger.info(
            f"Cross-validation completed. Mean ATE: {cv_summary['ate_mean']:.4f} ± {cv_summary['ate_std']:.4f}"
        )
        return self.results

    def _evaluate_fold(
        self, test_data: List[Dict[str, torch.Tensor]], fold_idx: int
    ) -> Dict[str, Any]:
        """Evaluate a single fold."""
        fold_ate_values = []
        fold_rpe_values = []

        for i, sequence_data in enumerate(test_data):
            try:
                predictions = self._predict_sequence(sequence_data)
                gt_poses = sequence_data["poses"]

                traj_metrics = compute_trajectory_metrics(
                    gt_poses.numpy(), predictions.numpy()
                )

                fold_ate_values.append(traj_metrics.ate_mean)
                fold_rpe_values.append(traj_metrics.rpe_trans_mean)

            except Exception as e:
                logger.error(f"Failed to evaluate sequence {i} in fold {fold_idx}: {e}")
                continue

        return {
            "fold_idx": fold_idx,
            "num_test_sequences": len(test_data),
            "ate_mean": np.mean(fold_ate_values),
            "ate_std": np.std(fold_ate_values),
            "rpe_mean": np.mean(fold_rpe_values),
            "rpe_std": np.std(fold_rpe_values),
        }

    def _aggregate_cv_results(
        self, fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate cross-validation results across folds."""
        ate_means = [fold["ate_mean"] for fold in fold_results]
        ate_stds = [fold["ate_std"] for fold in fold_results]
        rpe_means = [fold["rpe_mean"] for fold in fold_results]

        return {
            "ate_mean": np.mean(ate_means),
            "ate_std": np.std(ate_means),
            "ate_cv_score": np.std(ate_means)
            / np.mean(ate_means),  # Coefficient of variation
            "rpe_mean": np.mean(rpe_means),
            "rpe_std": np.std(rpe_means),
            "fold_ate_means": ate_means,
            "fold_ate_stds": ate_stds,
            "best_fold": np.argmin(ate_means),
            "worst_fold": np.argmax(ate_means),
        }


class BenchmarkEvaluator(BaseEvaluator):
    """Evaluator for standardized benchmarks."""

    BENCHMARK_CONFIGS = {
        "standard": {
            "metrics": ["ate", "rpe", "orientation_error"],
            "align_trajectories": True,
            "rpe_deltas": [1, 5, 10],
        },
        "realtime": {
            "metrics": ["ate", "rpe", "fps"],
            "align_trajectories": False,
            "measure_timing": True,
        },
        "robust": {
            "metrics": ["ate", "rpe", "completion_ratio"],
            "align_trajectories": True,
            "failure_threshold": 10.0,  # meters
        },
    }

    def evaluate(
        self,
        dataset: BaseIMUDataset,
        benchmark_type: str = "standard",
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model using standardized benchmark protocols.

        Args:
            dataset: Dataset to evaluate on
            benchmark_type: Type of benchmark ('standard', 'realtime', 'robust')
            custom_config: Custom benchmark configuration

        Returns:
            Dictionary containing benchmark results
        """
        if benchmark_type not in self.BENCHMARK_CONFIGS:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")

        config = self.BENCHMARK_CONFIGS[benchmark_type].copy()
        if custom_config:
            config.update(custom_config)

        logger.info(f"Starting {benchmark_type} benchmark evaluation")

        # Run trajectory evaluation with benchmark config
        trajectory_eval = TrajectoryEvaluator(self.model, self.device)
        results = trajectory_eval.evaluate(
            dataset,
            align_trajectories=config.get("align_trajectories", True),
            save_predictions=config.get("save_predictions", False),
        )

        # Add benchmark-specific analysis
        benchmark_analysis = self._analyze_benchmark_results(results, config)

        self.results = {
            "benchmark_type": benchmark_type,
            "benchmark_config": config,
            "evaluation_results": results,
            "benchmark_analysis": benchmark_analysis,
            "compliance": self._check_benchmark_compliance(benchmark_analysis, config),
        }

        return self.results

    def _analyze_benchmark_results(
        self, results: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze results according to benchmark criteria."""
        analysis = {}

        aggregate_metrics = results["aggregate_metrics"]

        # Standard metrics analysis
        if "ate" in config.get("metrics", []):
            analysis["ate_analysis"] = {
                "mean": aggregate_metrics["ate_mean"],
                "median": aggregate_metrics["ate_median"],
                "passes_threshold": aggregate_metrics["ate_mean"]
                < config.get("ate_threshold", 5.0),
            }

        if "fps" in config.get("metrics", []):
            analysis["performance_analysis"] = {
                "avg_fps": results["dataset_info"]["avg_fps"],
                "realtime_capable": results["dataset_info"]["avg_fps"]
                >= config.get("min_fps", 10.0),
            }

        if "completion_ratio" in config.get("metrics", []):
            failure_threshold = config.get("failure_threshold", 10.0)
            sequence_results = results["sequence_results"]

            failed_sequences = sum(
                1
                for seq in sequence_results
                if seq["trajectory"]["ate_max"] > failure_threshold
            )

            analysis["robustness_analysis"] = {
                "failed_sequences": failed_sequences,
                "success_rate": (len(sequence_results) - failed_sequences)
                / len(sequence_results),
                "robust": failed_sequences / len(sequence_results)
                < config.get("max_failure_rate", 0.1),
            }

        return analysis

    def _check_benchmark_compliance(
        self, analysis: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check if results comply with benchmark requirements."""
        compliance = {}

        if "ate_analysis" in analysis:
            compliance["ate_compliant"] = analysis["ate_analysis"]["passes_threshold"]

        if "performance_analysis" in analysis:
            compliance["performance_compliant"] = analysis["performance_analysis"][
                "realtime_capable"
            ]

        if "robustness_analysis" in analysis:
            compliance["robustness_compliant"] = analysis["robustness_analysis"][
                "robust"
            ]

        compliance["overall_compliant"] = all(compliance.values())

        return compliance
