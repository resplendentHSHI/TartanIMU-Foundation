"""
IMU noise utilities for neural inertial tracking.
Placeholder implementations for testing.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def add_realsense_noise(imu_data, noise_params=None):
    """
    Add RealSense IMU noise to the data.

    Args:
        imu_data: IMU data tensor
        noise_params: Noise parameters dictionary

    Returns:
        torch.Tensor: Noisy IMU data
    """
    logger.warning("add_realsense_noise: Placeholder implementation")
    return imu_data


def add_gyroscope_noise(gyro_data, noise_level=0.01):
    """
    Add gyroscope noise.

    Args:
        gyro_data: Gyroscope data
        noise_level: Noise level

    Returns:
        torch.Tensor: Noisy gyroscope data
    """
    if isinstance(gyro_data, torch.Tensor):
        noise = torch.randn_like(gyro_data) * noise_level
        return gyro_data + noise
    else:
        noise = np.random.normal(0, noise_level, gyro_data.shape)
        return gyro_data + noise


def add_accelerometer_noise(accel_data, noise_level=0.01):
    """
    Add accelerometer noise.

    Args:
        accel_data: Accelerometer data
        noise_level: Noise level

    Returns:
        torch.Tensor: Noisy accelerometer data
    """
    if isinstance(accel_data, torch.Tensor):
        noise = torch.randn_like(accel_data) * noise_level
        return accel_data + noise
    else:
        noise = np.random.normal(0, noise_level, accel_data.shape)
        return accel_data + noise
