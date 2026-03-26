#!/usr/bin/env python3
"""
Process raw iOS LaMAR data into npz format for TartanIMU training.

Reads from: raw_data/raw_ios/CAB/sessions/<session_id>/
Outputs to: data/new_lamar_split/human/{train,val}/CAB_<session_id>.npz

Each npz contains:
  retargetted_ts:   (N,)   timestamps in seconds, starting at 0, at 200Hz
  retargetted_imu:  (N, 6) [ax, ay, az, gx, gy, gz] in m/s^2 and rad/s
  retargetted_pos:  (N, 3) [x, y, z] in meters
  retargetted_quat: (N, 4) [qx, qy, qz, qw] (xyzw convention)

Key fixes over alignment_original.py:
  - iOS accelerometer reports in g's -> multiply by 9.81 for m/s^2
  - iOS gyroscope reports in rad/s (no conversion needed)
  - Timestamps are in nanoseconds
  - trajectories.txt quaternions are wxyz -> reorder to xyzw for npz
  - No gravity rotation applied (the dataloader handles gravity compensation)
  - SLERP interpolation for quaternions instead of linear
"""

import os
import sys
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

GRAVITY = 9.81  # m/s^2
TARGET_RATE = 200.0  # Hz


def read_accelerometer(filepath):
    """Read iOS accelerometer data. Returns (timestamps_ns, accel_xyz) in g's."""
    ts, ax, ay, az = [], [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts.append(float(parts[0].strip()))
            ax.append(float(parts[1].strip()))
            ay.append(float(parts[2].strip()))
            az.append(float(parts[3].strip()))
    return np.array(ts), np.column_stack([ax, ay, az])


def read_gyroscope(filepath):
    """Read iOS gyroscope data. Returns (timestamps_ns, gyro_xyz) in rad/s."""
    ts, rx, ry, rz = [], [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts.append(float(parts[0].strip()))
            rx.append(float(parts[1].strip()))
            ry.append(float(parts[2].strip()))
            rz.append(float(parts[3].strip()))
    return np.array(ts), np.column_stack([rx, ry, rz])


def read_trajectories(filepath):
    """Read trajectory data. Returns (timestamps_ns, quat_wxyz, pos_xyz)."""
    ts, quats, positions = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts.append(float(parts[0].strip()))
            # Format: timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covar
            qw = float(parts[2].strip())
            qx = float(parts[3].strip())
            qy = float(parts[4].strip())
            qz = float(parts[5].strip())
            tx = float(parts[6].strip())
            ty = float(parts[7].strip())
            tz = float(parts[8].strip())
            quats.append([qw, qx, qy, qz])
            positions.append([tx, ty, tz])
    return np.array(ts), np.array(quats), np.array(positions)


def process_session(session_dir, output_path):
    """Process a single iOS session into an npz file."""
    session_dir = Path(session_dir)
    session_name = session_dir.name

    accel_file = session_dir / 'accelerometer.txt'
    gyro_file = session_dir / 'gyroscope.txt'
    traj_file = session_dir / 'trajectories.txt'

    if not all(f.exists() for f in [accel_file, gyro_file, traj_file]):
        print(f"  SKIP {session_name}: missing required files")
        return False

    # Read raw data
    accel_ts, accel_data = read_accelerometer(accel_file)
    gyro_ts, gyro_data = read_gyroscope(gyro_file)
    traj_ts, traj_quat_wxyz, traj_pos = read_trajectories(traj_file)

    if len(accel_ts) < 100 or len(gyro_ts) < 100 or len(traj_ts) < 100:
        print(f"  SKIP {session_name}: too few samples (accel={len(accel_ts)}, gyro={len(gyro_ts)}, traj={len(traj_ts)})")
        return False

    # Convert accel from g's to m/s^2
    accel_data_ms2 = accel_data * GRAVITY

    # Find overlap region across all three signals
    overlap_start = max(accel_ts[0], gyro_ts[0], traj_ts[0])
    overlap_end = min(accel_ts[-1], gyro_ts[-1], traj_ts[-1])

    if overlap_end <= overlap_start:
        print(f"  SKIP {session_name}: no temporal overlap")
        return False

    duration = (overlap_end - overlap_start) / 1e9  # convert ns to seconds
    if duration < 10.0:
        print(f"  SKIP {session_name}: overlap too short ({duration:.1f}s)")
        return False

    # Create uniform 200Hz timestamps in the overlap region
    n_samples = int(duration * TARGET_RATE)
    common_ts_ns = np.linspace(overlap_start, overlap_end, n_samples)
    common_ts_s = (common_ts_ns - common_ts_ns[0]) / 1e9  # seconds from 0

    # Interpolate accelerometer (linear)
    accel_interp = np.zeros((n_samples, 3))
    for i in range(3):
        f = interp1d(accel_ts, accel_data_ms2[:, i], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
        accel_interp[:, i] = f(common_ts_ns)

    # Interpolate gyroscope (linear)
    gyro_interp = np.zeros((n_samples, 3))
    for i in range(3):
        f = interp1d(gyro_ts, gyro_data[:, i], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
        gyro_interp[:, i] = f(common_ts_ns)

    # Interpolate position (linear)
    pos_interp = np.zeros((n_samples, 3))
    for i in range(3):
        f = interp1d(traj_ts, traj_pos[:, i], kind='linear',
                      bounds_error=False, fill_value='extrapolate')
        pos_interp[:, i] = f(common_ts_ns)

    # Interpolate quaternions using SLERP
    # Convert wxyz -> xyzw for scipy Rotation
    traj_quat_xyzw = traj_quat_wxyz[:, [1, 2, 3, 0]]
    # Normalize quaternions
    norms = np.linalg.norm(traj_quat_xyzw, axis=1, keepdims=True)
    traj_quat_xyzw = traj_quat_xyzw / norms

    rotations = Rotation.from_quat(traj_quat_xyzw)
    slerp = Slerp(traj_ts, rotations)
    quat_interp = slerp(common_ts_ns).as_quat()  # xyzw convention

    # Combine IMU: [ax, ay, az, gx, gy, gz]
    imu = np.hstack([accel_interp, gyro_interp])

    # Save
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        retargetted_ts=common_ts_s,
        retargetted_imu=imu,
        retargetted_pos=pos_interp,
        retargetted_quat=quat_interp,
    )

    print(f"  OK {session_name}: {n_samples} samples ({duration:.1f}s) -> {output_path}")
    return True


def main():
    raw_dir = Path('raw_data/raw_ios/CAB/sessions')
    train_dir = Path('data/new_lamar_split/human/train')
    val_dir = Path('data/new_lamar_split/human/val')

    if not raw_dir.exists():
        print(f"ERROR: raw data dir not found: {raw_dir}")
        sys.exit(1)

    sessions = sorted(os.listdir(raw_dir))
    print(f"Found {len(sessions)} raw sessions")

    # Identify which sessions are in train vs val by checking existing symlinks/files
    train_sessions = set()
    val_sessions = set()
    for f in os.listdir(train_dir):
        if f.endswith('.npz') and f.startswith('CAB_'):
            session_name = f.replace('CAB_', '').replace('.npz', '')
            train_sessions.add(session_name)
    for f in os.listdir(val_dir):
        if f.endswith('.npz') and f.startswith('CAB_'):
            session_name = f.replace('CAB_', '').replace('.npz', '')
            val_sessions.add(session_name)

    print(f"Train CAB sessions: {len(train_sessions)}")
    print(f"Val CAB sessions: {len(val_sessions)}")

    success, fail = 0, 0
    for session_name in sessions:
        session_path = raw_dir / session_name

        if session_name in train_sessions:
            out_path = train_dir / f'CAB_{session_name}.npz'
        elif session_name in val_sessions:
            out_path = val_dir / f'CAB_{session_name}.npz'
        else:
            print(f"  SKIP {session_name}: not in train or val split")
            continue

        # Remove old symlink/file if exists
        if out_path.is_symlink():
            out_path.unlink()
        elif out_path.exists():
            out_path.unlink()

        ok = process_session(session_path, out_path)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} processed, {fail} failed/skipped")


if __name__ == '__main__':
    main()
