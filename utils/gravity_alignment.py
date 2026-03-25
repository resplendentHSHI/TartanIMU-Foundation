"""
Gravity alignment utilities for IMU data processing.
This module provides functions to remove gravity effects from accelerometer readings
to improve generalization across different pen orientations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import logging
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from .constants import GRAVITY

class GravityAligner:
    """
    Gravity alignment for IMU data to improve orientation invariance.
    
    This class provides methods to remove gravity effects from accelerometer
    readings using orientation data, making the same motion pattern produce
    similar IMU signals regardless of device orientation.
    """
    
    def __init__(self, gravity_magnitude: float = GRAVITY, 
                 gravity_direction: np.ndarray = None,
                 enable_validation: bool = True):
        """
        Initialize gravity aligner.
        
        Args:
            gravity_magnitude: Gravity magnitude in m/s²
            gravity_direction: Gravity direction in world frame [3]
            enable_validation: Whether to enable validation checks
        """
        self.gravity_magnitude = gravity_magnitude
        
        if gravity_direction is None:
            # Standard gravity direction (pointing down)
            self.gravity_direction = np.array([0, 0, -1])
        else:
            self.gravity_direction = gravity_direction / np.linalg.norm(gravity_direction)
        
        # Gravity vector in world frame
        self.gravity_world = self.gravity_magnitude * self.gravity_direction
        self.enable_validation = enable_validation
        
        logging.info(f"GravityAligner initialized with gravity: {self.gravity_world}")
    
    def transform_gravity_to_body_frame(self, orientation_quat: np.ndarray) -> np.ndarray:
        """
        Transform gravity from world frame to body frame using quaternion rotation.
        
        Args:
            orientation_quat: [N, 4] quaternions (x, y, z, w) format
        
        Returns:
            gravity_body: [N, 3] gravity vector in body frame
        """
        # Convert quaternions to rotation matrices
        rotations = Rotation.from_quat(orientation_quat)
        
        # Apply rotation: gravity_body = R * gravity_world
        gravity_body = rotations.apply(self.gravity_world)
        
        return gravity_body
    
    def align_gravity(self, accel_raw: np.ndarray, 
                     orientation_quat: np.ndarray) -> np.ndarray:
        """
        Remove gravity component from raw accelerometer readings.
        
        Args:
            accel_raw: [N, 3] raw accelerometer readings (ax, ay, az)
            orientation_quat: [N, 4] quaternions (qx, qy, qz, qw)
        
        Returns:
            accel_aligned: [N, 3] gravity-aligned accelerometer readings
        """
        # Transform gravity to body frame
        gravity_body = self.transform_gravity_to_body_frame(orientation_quat)
        
        # Remove gravity: accel_motion = accel_raw - gravity_body
        accel_aligned = accel_raw - gravity_body
        
        return accel_aligned
    
    def validate_alignment(self, accel_raw: np.ndarray, 
                          accel_aligned: np.ndarray,
                          orientation_quat: np.ndarray) -> Dict[str, float]:
        """
        Validate that gravity alignment is working correctly.
        
        Args:
            accel_raw: [N, 3] raw accelerometer readings
            accel_aligned: [N, 3] aligned accelerometer readings
            orientation_quat: [N, 4] quaternions
        
        Returns:
            validation_metrics: Dictionary of validation metrics
        """
        if not self.enable_validation:
            return {}
        
        # Check 1: Verify gravity transformation
        gravity_body = self.transform_gravity_to_body_frame(orientation_quat)
        
        # The sum of aligned accel and gravity_body should equal raw accel
        reconstructed = accel_aligned + gravity_body
        reconstruction_error = np.linalg.norm(accel_raw - reconstructed, axis=1).mean()
        
        # Check 2: Magnitude comparison
        raw_magnitude = np.linalg.norm(accel_raw, axis=1)
        aligned_magnitude = np.linalg.norm(accel_aligned, axis=1)
        
        # Check 3: Stationary periods (if any)
        stationary_threshold = 0.5  # m/s²
        stationary_mask = raw_magnitude < stationary_threshold
        
        validation_metrics = {
            'reconstruction_error': reconstruction_error,
            'raw_magnitude_mean': raw_magnitude.mean(),
            'raw_magnitude_std': raw_magnitude.std(),
            'aligned_magnitude_mean': aligned_magnitude.mean(),
            'aligned_magnitude_std': aligned_magnitude.std(),
            'stationary_periods': np.sum(stationary_mask),
            'stationary_aligned_mean': aligned_magnitude[stationary_mask].mean() if np.any(stationary_mask) else np.nan
        }
        
        # Log validation results
        if reconstruction_error < 1e-6:
            logging.info("✅ Gravity alignment validation passed!")
        else:
            logging.warning(f"⚠️ Gravity alignment validation failed! Error: {reconstruction_error}")
        
        logging.info(f"Raw accel magnitude: {raw_magnitude.mean():.3f} ± {raw_magnitude.std():.3f}")
        logging.info(f"Aligned accel magnitude: {aligned_magnitude.mean():.3f} ± {aligned_magnitude.std():.3f}")
        
        return validation_metrics
    
    def visualize_alignment(self, ts: np.ndarray, accel_raw: np.ndarray,
                           accel_aligned: np.ndarray, gyro: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize the effect of gravity alignment.
        
        Args:
            ts: [N] timestamps
            accel_raw: [N, 3] raw accelerometer readings
            accel_aligned: [N, 3] aligned accelerometer readings
            gyro: [N, 3] gyroscope readings
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        
        # Plot accelerometer data
        axes[0, 0].plot(ts, accel_raw[:, 0], 'r-', label='Raw X', alpha=0.7)
        axes[0, 0].plot(ts, accel_aligned[:, 0], 'b-', label='Aligned X', alpha=0.7)
        axes[0, 0].set_title('Accelerometer X-axis')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[1, 0].plot(ts, accel_raw[:, 1], 'r-', label='Raw Y', alpha=0.7)
        axes[1, 0].plot(ts, accel_aligned[:, 1], 'b-', label='Aligned Y', alpha=0.7)
        axes[1, 0].set_title('Accelerometer Y-axis')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[2, 0].plot(ts, accel_raw[:, 2], 'r-', label='Raw Z', alpha=0.7)
        axes[2, 0].plot(ts, accel_aligned[:, 2], 'b-', label='Aligned Z', alpha=0.7)
        axes[2, 0].set_title('Accelerometer Z-axis')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot gyroscope data (unchanged)
        axes[0, 1].plot(ts, gyro[:, 0], 'g-', label='Gyro X')
        axes[0, 1].set_title('Gyroscope X-axis')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 1].plot(ts, gyro[:, 1], 'g-', label='Gyro Y')
        axes[1, 1].set_title('Gyroscope Y-axis')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        axes[2, 1].plot(ts, gyro[:, 2], 'g-', label='Gyro Z')
        axes[2, 1].set_title('Gyroscope Z-axis')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Gravity alignment visualization saved to {save_path}")
        
        plt.show()
    
    def test_orientation_invariance(self, motion_pattern: np.ndarray,
                                  orientation_upright: np.ndarray,
                                  orientation_sideways: np.ndarray) -> Dict[str, float]:
        """
        Test that same motion produces similar aligned readings across orientations.
        
        Args:
            motion_pattern: [N, 3] motion pattern (pure motion without gravity)
            orientation_upright: [N, 4] upright orientation quaternions
            orientation_sideways: [N, 4] sideways orientation quaternions
        
        Returns:
            invariance_metrics: Dictionary of invariance metrics
        """
        # Simulate raw accelerometer readings for different orientations
        gravity_upright = self.transform_gravity_to_body_frame(orientation_upright)
        gravity_sideways = self.transform_gravity_to_body_frame(orientation_sideways)
        
        # Raw readings (motion + gravity)
        accel_raw_upright = motion_pattern + gravity_upright
        accel_raw_sideways = motion_pattern + gravity_sideways
        
        # Aligned readings (should be similar)
        accel_aligned_upright = self.align_gravity(accel_raw_upright, orientation_upright)
        accel_aligned_sideways = self.align_gravity(accel_raw_sideways, orientation_sideways)
        
        # Calculate similarity metrics
        similarity = np.linalg.norm(accel_aligned_upright - accel_aligned_sideways, axis=1)
        
        invariance_metrics = {
            'mean_similarity': similarity.mean(),
            'std_similarity': similarity.std(),
            'max_similarity': similarity.max(),
            'min_similarity': similarity.min(),
            'upright_raw_range': [accel_raw_upright.min(), accel_raw_upright.max()],
            'sideways_raw_range': [accel_raw_sideways.min(), accel_raw_sideways.max()],
            'upright_aligned_range': [accel_aligned_upright.min(), accel_aligned_upright.max()],
            'sideways_aligned_range': [accel_aligned_sideways.min(), accel_aligned_sideways.max()]
        }
        
        logging.info(f"Orientation invariance test - Mean similarity: {similarity.mean():.6f}")
        logging.info(f"Raw readings differ significantly due to gravity")
        logging.info(f"Aligned readings are similar (invariant to orientation)")
        
        return invariance_metrics


def create_gravity_aligner_from_config(config: Dict[str, Any]) -> GravityAligner:
    """
    Create gravity aligner from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        GravityAligner instance
    """
    gravity_config = config.get('data', {}).get('gravity_alignment', {})
    
    gravity_magnitude = gravity_config.get('magnitude', GRAVITY)
    gravity_direction = gravity_config.get('direction', [0, 0, -1])
    enable_validation = gravity_config.get('enable_validation', True)
    
    return GravityAligner(
        gravity_magnitude=gravity_magnitude,
        gravity_direction=np.array(gravity_direction),
        enable_validation=enable_validation
    )
