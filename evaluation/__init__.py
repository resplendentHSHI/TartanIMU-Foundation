"""
Evaluation module for neural inertial tracking.

This module provides evaluation utilities, metrics, and evaluators
for assessing model performance on various tasks.
"""

from . import postprocess
from .evaluators import (
    BaseEvaluator,
    BenchmarkEvaluator,
    CrossValidationEvaluator,
    SequenceEvaluator,
    TrajectoryEvaluator,
)
from .metrics import (
    PoseMetrics,
    SequenceMetrics,
    TrajectoryMetrics,
    compute_accruacy_metrics,
    compute_ate,
    compute_ate_rte,
    compute_orientation_error,
    compute_rpe,
    compute_translation_error,
)

__all__ = [
    # Metrics
    "TrajectoryMetrics",
    "PoseMetrics",
    "SequenceMetrics",
    "compute_ate",
    "compute_rpe",
    "compute_orientation_error",
    "compute_translation_error",
    "compute_ate_rte",
    "compute_accruacy_metrics",
    # Evaluators
    "BaseEvaluator",
    "TrajectoryEvaluator",
    "SequenceEvaluator",
    "CrossValidationEvaluator",
    "BenchmarkEvaluator",
    # Postprocess
    "postprocess",
]
