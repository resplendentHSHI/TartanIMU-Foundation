import logging
import os
import pdb
import random
from os import path as osp

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import torch
import transformations as tf
from numpy.random import normal as gen_normal
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset
from utils.gravity_alignment import create_gravity_aligner_from_config
from utils.imu_noise import add_realsense_noise

# import pypose as pp


def quart_to_euler(x, y, z, w):
    (roll, pitch, yaw) = tf.euler_from_quaternion([w, x, y, z])
    return [roll, pitch, yaw]


def euler_to_quart(roll, pitch, yaw):
    (w, x, y, z) = tf.quaternion_from_euler(roll, pitch, yaw)
    return [x, y, z, w]


class AirLabNPZSequence(object):
    def __init__(
        self,
        data_path,
        imu_freq,
        window_size,
        verbose=True,
        use_local_coord=False,
        mode="test",
        plot=False,
        gravity_aligner=None,
    ):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = (
            imu_freq  # FIXME: AirLab NPZ have a fixed imu frequency of 200HZ.
        )
        self.interval = window_size  # 100
        self.data_valid = False
        self.sum_duration = 0
        self.valid = False
        self.plot = plot
        self.add_noise = False
        self.mode = mode
        self.use_local_coord = use_local_coord
        self.motion_type = None
        self.gravity_aligner = gravity_aligner

        if data_path is not None:
            self.valid = self.load(data_path, verbose=verbose)

    def calculate_velocity_from_poses(self, ts, pos, quat):
        """
        Calculate velocity from pose data with safety checks.

        Args:
            ts (np.ndarray): Timestamps
            pos (np.ndarray): Positions
            quat (np.ndarray): Quaternions

        Returns:
            tuple: (velocity_global, velocity_body)
        """
        # Input validation
        if len(ts) < 2 or len(pos) < 2 or len(quat) < 2:
            raise ValueError("Input arrays must have at least 2 elements")

        if len(ts) != len(pos) or len(ts) != len(quat):
            raise ValueError("All input arrays must have the same length")

        # Calculate relative positions
        pos_gt = pos - pos[0]  # Ground truth positions relative to the starting point
        quat_gt = Rotation.from_quat(quat)

        # Calculate time differences
        dt = ts[1:] - ts[:-1]

        # Prevent division by zero and handle very small time differences
        dt = np.where(dt == 0, 1e-6, dt)
        dt = np.where(dt < 1e-9, 1e-6, dt)  # Handle extremely small values

        # Ensure proper shape for broadcasting
        if dt.shape == (dt.shape[0],):
            dt = dt[:, np.newaxis]  # (N, 1)

        # Calculate velocities
        velocity_global = (pos_gt[1:] - pos_gt[:-1]) / dt
        velocity_body = quat_gt[:-1].inv().apply(velocity_global)

        return velocity_global, velocity_body

    def load(self, data_path, verbose=False):

        # read data from npz file with error handling
        try:
            all_data = np.load(data_path)
            if verbose:
                logging.info(
                    f"Successfully loaded {data_path} with keys: {list(all_data.keys())}"
                )
        except (EOFError, OSError, ValueError) as e:
            logging.warning(
                f"[DATA_LOAD] Failed to load file {data_path}: {e}"
            )
            return False
        # Fix 4: Consistent tensor creation with proper dtype
        if "car" in data_path:
            self.motion_type = torch.tensor(1, dtype=torch.long)  # car
        elif "dog" in data_path:
            self.motion_type = torch.tensor(2, dtype=torch.long)  # dog
        elif "drone" in data_path:
            self.motion_type = torch.tensor(3, dtype=torch.long)  # drone
        elif "human" in data_path:
            self.motion_type = torch.tensor(4, dtype=torch.long)  # human
        else:
            self.motion_type = torch.tensor(0, dtype=torch.long)  #'no_label'

        ts = all_data["retargetted_ts"]
        imu = all_data["retargetted_imu"]
        pos = all_data["retargetted_pos"]
        quat = all_data["retargetted_quat"]  # in xyzw format

        if verbose:
            logging.info(
                f"Data shapes - ts: {ts.shape}, imu: {imu.shape}, pos: {pos.shape}, quat: {quat.shape}"
            )
        # quat /= np.linalg.norm(quat, axis=1)[:,None]
        # obtain basic data components: imu measurement and odometry result
        accel_body = imu[:, :3]
        angular_vel_body = imu[:, 3:]

        velocity_global, velocity_body = self.calculate_velocity_from_poses(
            ts, pos, quat
        )  # use the ground truth to calculate the velocity in global and local coordinate system

        # TODO: Only for Tartan Drive Dataset
        pos_gt = pos - pos[0]
        quat_gt = Rotation.from_quat(quat)
        self.quat_gt = quat_gt

        # transform body frame imu measurements into global coordinate for easier learning
        # this requires rotation from body to world frame. There are two options to get it:
        #
        #   1. use ground truth rotation from vio supervision signal
        #   2. (pre)integrate imu data and obtain a approximate

        # Gravity compensation using GravityAligner if available
        if self.gravity_aligner is not None:
            # Use the new gravity alignment approach
            accel_aligned = self.gravity_aligner.align_gravity(accel_body, quat)

            if not self.use_local_coord:
                # Transform to global frame
                accel_est = global_to_local_rotation.apply(accel_aligned)
                angular_vel_est = global_to_local_rotation.apply(angular_vel_body)
            else:
                # Keep in body frame
                accel_est = accel_aligned
                angular_vel_est = angular_vel_body

            # Validate alignment if enabled
            if self.gravity_aligner.enable_validation:
                validation_metrics = self.gravity_aligner.validate_alignment(
                    accel_body, accel_aligned, quat
                )
                if verbose:
                    logging.info(f"Gravity alignment validation: {validation_metrics}")
        else:
            # Fallback to original gravity compensation
            gravity = 9.8101
            global_to_local_rotation = quat_gt

            if not self.use_local_coord:
                # case 1: use ground truth rotation from vio supervision signal
                global_to_local_rotation = quat_gt
                accel_est = global_to_local_rotation.apply(accel_body)
                accel_est -= np.array([0, 0, gravity])
                angular_vel_est = global_to_local_rotation.apply(angular_vel_body)
            else:
                # case 2: using IMU in body coord
                #@ Dataloader Mod Here
                accel_est = accel_body 
                - global_to_local_rotation.inv().apply(
                    np.array([0, 0, gravity])
                )
                angular_vel_est = angular_vel_body

        # Fix 3: Add bounds checking for interval slicing
        if self.interval >= len(pos_gt):
            logging.warning(
                f"[DATA_LOAD] Rejecting {data_path}: interval {self.interval} >= sequence length {len(pos_gt)}"
            )
            return False

        gt_disp = (
            pos_gt[self.interval :] - pos_gt[: -self.interval]
        )  # Get the ground truth position offset corresponding to each window_size

        self.seq_len = ts.shape[0]
        if self.seq_len < self.imu_freq * 10:
            logging.warning(
                f"[DATA_LOAD] Rejecting {data_path}: seq_len {self.seq_len} < imu_freq*10 ({self.imu_freq*10})"
            )
            return False
        self.dt = ts[1] - ts[0]  # ts already clean

        self.get_gt = False
        self.data_valid = True
        self.sum_duration = ts[-1] - ts[0]

        if verbose:
            logging.info(
                f"{data_path}: sum time: {self.sum_duration}, gt: {self.get_gt}"
            )

        new_ts = ts[:, np.newaxis]
        self.ts = new_ts
        # self.features = np.concatenate([glob_gyro, glob_acce, tmp_gyro, tmp_acce], axis=1)
        self.features = np.concatenate(
            [angular_vel_est, accel_est], axis=1
        )  # global gyro and accel
        self.orientations = quat_gt.as_quat()
        self.pos_gt = pos_gt
        self.gt_ori = quat_gt.as_quat()
        if not self.use_local_coord:
            self.targets = gt_disp
        else:
            local_disp = quat_gt[self.interval :].inv().apply(
                pos_gt[self.interval :]
            ) - quat_gt[: -self.interval].inv().apply(
                pos_gt[:-200]
            )  # Displacement difference in local coordinate system for every 200 frames of GT
            # self.targets = local_disp
            # cum_sum = np.cumsum(velocity_body, axis=0)
            # vel_body_mean = (cum_sum[self.interval:] - cum_sum[:-self.interval]) / self.interval
            # front_means = cum_sum[:self.interval] / np.arange(1, self.interval + 1).reshape(-1, 1)
            # vel_body_mean = np.vstack([front_means, vel_body_mean])
            num_windows = velocity_body.shape[0] - self.interval + 1
            vel_body_mean = np.zeros((num_windows, velocity_body.shape[1]))
            for i in range(num_windows):
                window = velocity_body[i : i + self.interval]
                vel_body_mean[i] = np.mean(window, axis=0, keepdims=True)
            self.targets = vel_body_mean

        self.ts = self.ts[:-1]  # N*3
        self.features = self.features[:-1]  # N*6
        self.orientations = self.orientations[:-1]  # N*4
        self.pos_gt = self.pos_gt[:-1]  # N*3
        self.gt_ori = self.gt_ori[:-1]  # N*4
        self.targets = self.targets[:-1]  # (N-200)*3
        self.velocity_body = velocity_body  # N*3
        """
        # angle = [] for debug
        # init_angle = gt_rot_angle[10000]
        # for i in range(10000,10020):
        #     angle.append((init_angle+glob_gyro[i] * 0.005).unsqueeze(0))
        #     init_angle = init_angle+glob_gyro[i] * 0.005
        # inte_angle = torch.cat(angle, dim=0)
        # inte_angle = torch.tensor([(gt_rot_angle[i]]+glob_gyro[i]*0.005) for i in range(0,20)])
        # dr = pp.so3(torch.tensor(glob_gyro)*self.dt).Exp()
        # incre_r = pp.cumprod(dr, dim = 0, left=False)
        # euler_test = torch.tensor(glob_gyro[10001:10011])*self.dt[10001:10011]
        # quan_test=torch.tensor([euler_to_quart(euler_test[i,0], euler_test[i,1], euler_test[i,2]) for i in range(len(euler_test))])
        """
        return True

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_data_valid(self):
        return self.data_valid

    def get_aux(self):
        return np.concatenate(
            [self.ts.reshape(-1, 1), self.orientations, self.pos_gt, self.gt_ori],
            axis=1,
        )


class BasicSequenceData(object):
    def __init__(self, cfg, source_folder, verbose=False, **kwargs):
        super(BasicSequenceData, self).__init__()
        self.window_size = int(
            cfg["model_param"]["window_time"] * cfg["data"]["imu_freq"]
        )  # 100
        self.past_data_size = int(
            cfg["model_param"]["past_time"] * cfg["data"]["imu_freq"]
        )  # 0
        self.future_data_size = int(
            cfg["model_param"]["future_time"] * cfg["data"]["imu_freq"]
        )  # 0
        self.step_size = int(cfg["data"]["imu_freq"] / cfg["data"]["sample_freq"])  # 5
        self.seq_len = cfg["train"]["seq_len"]  # 10
        self.add_noise = cfg["train"]["add_noise"]
        self.index_map = []
        self.ts, self.gt_pos, self.gt_ori = [], [], []
        self.features, self.targets = [], []
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1
        self.use_local_coord = cfg["data"]["use_local_coord"]
        self.mode = kwargs.get("mode", "train")

        # Create gravity aligner if enabled
        self.gravity_aligner = None
        if cfg["data"].get("gravity_alignment", {}).get("enabled", False):
            try:
                self.gravity_aligner = create_gravity_aligner_from_config(cfg)
                if verbose:
                    logging.info("✅ Gravity alignment enabled")
            except Exception as e:
                logging.warning(f"Failed to create gravity aligner: {e}")
                self.gravity_aligner = None

        sum_t = 0
        win_dt = self.window_size / cfg["data"]["imu_freq"]  # 1.0
        self.length = kwargs.get("current_frame")
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        max_v_norm = (
            5.0  # raw: 10 fixed for drone, 65 for tartandrive, 100 for local velocity
        )
        valid_i = 0

        # parse datalist from source folder

        data_list = source_folder
        logging.info(f"[DATA_LOAD] Starting load of {len(data_list)} files (mode={self.mode})")

        for i in range(len(data_list)):
            data_path = data_list[i]
            logging.info(f"[DATA_LOAD] Loading file {i+1}/{len(data_list)}: {data_path}")
            try:
                seq = AirLabNPZSequence(
                    data_path,
                    cfg["data"]["imu_freq"],
                    self.window_size,
                    verbose=True,
                    use_local_coord=self.use_local_coord,
                    mode=self.mode,
                    gravity_aligner=self.gravity_aligner,
                )

                if seq.valid is False:
                    logging.warning(f"[DATA_LOAD] Skipping invalid file (seq.valid=False): {data_path}")
                    continue
            except Exception as e:
                logging.warning(f"[DATA_LOAD] Failed to process file {data_path}: {e}", exc_info=True)
                continue
            # feat:  The data of the whole trajectory is the IMU data, and the data of the first 100 frames is the difference between the first and last positions of the GT pose.
            feat, targ, aux, motion_type = (
                seq.get_feature(),
                seq.get_target(),
                seq.get_aux(),
                seq.motion_type,
            )  # feature, targ is the concatenation of timestamps, rotations, and the rotation and translation between each timestamp

            if self.length is not None:
                feat = feat[: self.length, :]
                targ = targ[: (self.length - int(cfg["data"]["imu_freq"])), :]
                aux = aux[: self.length, :]

            sum_t += seq.sum_duration
            valid_samples = 0
            index_map = []
            step_size = self.step_size  # 5

            if self.mode in ["train", "val", "test"] and seq.get_gt is False:
                for j in range(
                    self.past_data_size,
                    targ.shape[0]
                    - self.future_data_size
                    - (self.seq_len - 1) * self.window_size,  #
                    step_size,
                ):  # 0-8000 , step_size=10
                    outlier = False
                    for k in range(
                        self.seq_len
                    ):  # 10 0-9  is the number of sliding windows, each window time length is 1S, and the IMU frequency is 200HZ.
                        index = (
                            j + k * self.window_size
                        )  # j=0: index=0,100,200,..900, j=10: index=10,110,120,..910
                        velocity = np.linalg.norm(targ[index] / win_dt)
                        if velocity > max_v_norm:  # max_v_norm=4.0
                            # If the norm of the velocity in the xyz three axes in each window is greater than 4, it is considered an outlier.
                            outlier = True
                            break
                    if outlier is False:
                        index_map.append(
                            [valid_i, j, motion_type]
                        )  # valid_i is the id of the trajectory, j is the front end of each window of length 10*imu_freq in the current trajectory
                        self.valid_all_samples += 1
                        valid_samples += 1
            else:
                for j in range(
                    self.past_data_size,
                    targ.shape[0]
                    - self.future_data_size
                    - (self.seq_len - 1) * self.window_size,
                    step_size,
                ):  # (0,length-900,5)
                    index_map.append([valid_i, j, motion_type])
                    self.valid_all_samples += 1
                    valid_samples += 1
            if len(index_map) > 0:
                self.data_paths.append(data_list[i])  # add data path
                self.index_map.append(index_map)
                self.features.append(feat)  # feature
                self.targets.append(targ)  # position offset between each timestamp
                self.ts.append(aux[:, 0])  # timestamps
                # self.orientations.append(aux[:, 1:5]) #rotation
                self.gt_pos.append(aux[:, 5:8])  # translation
                self.gt_ori.append(aux[:, 8:12])  # rotation
                self.valid_samples.append(
                    valid_samples
                )  # the number of valid samples in the interval of 5 sampling
                valid_i += 1
                self.motion_type = motion_type
        if verbose:
            logging.info(f"datasets sum time {sum_t}")

        # Check if we have any valid data
        if len(self.data_paths) == 0:
            raise ValueError(
                f"No valid data files found in the provided paths. Please check your data directory."
            )

    def plot_targets(self, out_dir):
        targets = np.concatenate(self.targets, axis=0)
        targets = np.clip(targets, -1.0, 1.0)
        fig = plt.figure(num="targets", dpi=90, figsize=(9, 9))
        plt.hist2d(
            targets[:, 0],
            targets[:, 1],
            bins=100,
            norm=matplotlib.colors.LogNorm(),
            cmap=matplotlib.cm.jet,
        )
        plt.title("targets")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        fig.savefig(osp.join(out_dir, "targets.png"))
        plt.close("all")

        fig = plt.figure(num="targets", dpi=90, figsize=(9, 9))
        plt.hist(targets[:, 2], bins=100)
        plt.title("targets_z")
        fig.savefig(osp.join(out_dir, "targets_z.png"))
        plt.close("all")

        fig = plt.figure(num="targets_n", dpi=90, figsize=(9, 9))
        plt.hist(np.linalg.norm(targets[:, 0:2], axis=-1), bins=100)
        plt.title("targets_norm")
        fig.savefig(osp.join(out_dir, "targets_norm.png"))
        plt.close("all")

    def get_data(self):
        return (
            self.features,
            self.targets,
            self.ts,
            self.gt_pos,
            self.gt_ori,
            self.motion_type,
        )

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        index_map = []
        for i in range(len(self.index_map)):  # 0-240
            index_map += self.index_map[i]
        return index_map


class ResNetLSTMSeqToSeqDataset(Dataset):
    def __init__(self, cfg, basic_data: BasicSequenceData, index_map, **kwargs):
        super(ResNetLSTMSeqToSeqDataset, self).__init__()
        self.window_size = basic_data.window_size  # 100
        self.past_data_size = basic_data.past_data_size  # 0
        self.future_data_size = basic_data.future_data_size  # 0
        self.step_size = basic_data.step_size  # 5
        self.seq_len = basic_data.seq_len  # 10

        self.add_bias_noise = cfg["augment"]["add_bias_noise"]  # True
        self.accel_bias_range = cfg["augment"]["accel_bias_range"]  # 0.1
        self.gyro_bias_range = cfg["augment"]["gyro_bias_range"]  # 0.002
        if self.add_bias_noise is False:
            self.accel_bias_range = 0.0
            self.gyro_bias_range = 0.0
        self.add_gravity_noise = cfg["augment"]["add_gravity_noise"]  # True
        self.gravity_noise_theta_range = cfg["augment"][
            "gravity_noise_theta_range"
        ]  # 5

        self.feat_acc_sigma = cfg["augment"]["feat_acc_sigma"]  # 0.0001
        self.feat_gyr_sigma = cfg["augment"]["feat_gyr_sigma"]  # 1e-05
        self.use_local_coord = cfg["data"]["use_local_coord"]

        # Time-scaling invariance settings
        self.add_time_scaling = cfg["augment"].get("add_time_scaling", False)
        self.time_scale_range = cfg["augment"].get("time_scale_range", [0.5, 2.0])
        self.time_scaling_probability = cfg["augment"].get(
            "time_scaling_probability", 0.7
        )

        self.mode = kwargs.get("mode", "train")  #'train'
        self.shuffle, self.transform, self.gauss = False, False, False
        if self.mode == "train":
            self.shuffle = True
            self.transform = True
            self.gauss = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        (
            self.features,
            self.targets,
            self.ts,
            self.gt_pos,
            self.gt_ori,
            self.motion_type,
        ) = basic_data.get_data()
        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)

    def time_scale_augmentation(self, feat, targ, time_scale_range=(0.5, 2.0)):
        """
        Apply time-scaling invariance by warping IMU windows and scaling velocity targets.

        Args:
            feat: [T, 6] IMU data (accel + gyro)
            targ: [seq_len, 3] velocity targets
            time_scale_range: Range for time warping (stretch/compress)

        Returns:
            Tuple of (augmented_feat, augmented_targ)
        """
        from scipy.interpolate import interp1d

        # Random time scale factor
        time_scale = np.random.uniform(*time_scale_range)

        T = feat.shape[0]

        # Create original and warped time indices
        t_orig = np.linspace(0, 1, T)
        t_warped = np.linspace(0, 1, int(T * time_scale))

        # Interpolate IMU data
        augmented_feat = np.zeros_like(feat)
        for i in range(6):
            # Handle case where warped time is shorter than original
            if len(t_warped) <= T:
                # Pad with last value if needed
                feat_padded = np.pad(
                    feat[: len(t_warped), i],
                    (0, max(0, T - len(t_warped))),
                    mode="edge",
                )
                interp_func = interp1d(
                    t_warped,
                    feat_padded[: len(t_warped)],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                # Truncate if warped time is longer
                interp_func = interp1d(
                    t_warped[:T],
                    feat[:T, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )

            augmented_feat[:, i] = interp_func(t_orig)

        # Scale velocity targets inversely
        # This is the key insight: if we stretch time, velocity should be scaled down
        augmented_targ = targ / time_scale

        return augmented_feat, augmented_targ

    def __getitem__(self, item):
        seq_id, frame_id, label = (
            self.index_map[item][0],
            self.index_map[item][1],
            self.index_map[item][2],
        )  # the number of the sequence and the number of the frame
        # The length of the sequence is 1000 frames, which is 10 window_size.
        # from the framed_id position to take seq_len*window_size length of data
        # Angular velocity and linear acceleration in global/local coordinate system
        feat = self.features[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.seq_len * self.window_size
            + self.future_data_size
        ]  # 1000*6 Take past several frames, future several frames, and several frames in the current window
        # if self.use_local_coord: #velocity_body
        #     targ = self.targets[seq_id][frame_id : frame_id + self.seq_len * self.window_size: self.window_size]
        # else: #global pos diff

        targ = self.targets[seq_id][
            frame_id : frame_id + self.seq_len * self.window_size : self.window_size
        ]  # the beginning of the sequence 10*3 local coordinate system targ is the body velocity mean in 0-200 200-400 400-600
        ori = self.gt_ori[seq_id][
            frame_id : frame_id + self.seq_len * self.window_size
        ]  # 2000*4 gt orientation
        if self.mode in ["train"]:
            targ_aug = np.copy(targ)
            feat_aug = np.copy(feat)
            if self.transform:
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
                )
                feat_aug[:, 0:2] = np.matmul(rm, feat_aug[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat_aug[:, 3:5].T).T
                targ_aug[:, 0:2] = np.matmul(rm, targ_aug[:, 0:2].T).T

            if self.add_bias_noise:
                # shift in the accel and gyro bias terms
                random_bias = np.random.random((1, 6))
                random_bias[:, 0:3] = (
                    (random_bias[:, 0:3] - 0.5) * self.gyro_bias_range / 0.5
                )
                random_bias[:, 3:6] = (
                    (random_bias[:, 3:6] - 0.5) * self.accel_bias_range / 0.5
                )
                feat_aug += random_bias

            if self.add_gravity_noise:
                # Generate random rotation around a random axis in XY plane
                angle_rand = random.random() * np.pi * 2
                vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
                theta_rand = (
                    random.random() * np.pi * self.gravity_noise_theta_range / 180.0
                )
                rvec = theta_rand * vec_rand
                r = Rotation.from_rotvec(rvec)
                R_mat = r.as_matrix()

                # Apply rotation to IMU data (accelerometer and gyroscope)
                feat_aug[:, 0:3] = np.matmul(R_mat, feat_aug[:, 0:3].T).T  # Gyroscope
                feat_aug[:, 3:6] = np.matmul(
                    R_mat, feat_aug[:, 3:6].T
                ).T  # Accelerometer

                # For body-frame velocity targets, rotate targets too (SO(3)-equivariant)
                # This ensures physical consistency: rotated IMU → rotated body velocity
                if hasattr(self, "use_local_coord") and self.use_local_coord:
                    targ_aug = np.matmul(
                        R_mat, targ_aug.T
                    ).T  # Rotate body-frame velocity targets

            if self.gauss:
                if self.feat_gyr_sigma > 0:
                    feat_aug[:, 0:3] += gen_normal(
                        loc=0.0,
                        scale=self.feat_gyr_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    )
                if self.feat_acc_sigma > 0:
                    feat_aug[:, 3:6] += gen_normal(
                        loc=0.0,
                        scale=self.feat_acc_sigma,
                        size=(len(feat_aug[:, 0]), 3),
                    )

            # Apply time-scaling invariance augmentation
            if self.add_time_scaling:
                if np.random.random() < self.time_scaling_probability:
                    feat_aug, targ_aug = self.time_scale_augmentation(
                        feat_aug, targ_aug, self.time_scale_range
                    )

            feat = feat_aug
            targ = targ_aug

        seq_feat = []
        for i in range(self.seq_len):  # 0-9
            seq_feat.append(
                feat[
                    i * self.window_size : self.past_data_size
                    + (i + 1) * self.window_size
                    + self.future_data_size,
                    :,
                ].T
            )  # Divide feat into ten segments, 0-100, 100-200, 200-300

        # if isinstance(seq_feat, list):
        #     seq_feat = np.array(np.stack(seq_feat, axis=0))
        # else:
        try:
            seq_feat = np.array(seq_feat)
        except ValueError as e:
            print("ValueError:", e)

        #

        return (
            seq_feat.astype(np.float32),
            targ.astype(np.float32),
            ori.astype(np.float32),
            label,
        )  # 10*6*100  10*3

    def __len__(self):
        return len(self.index_map)


def SeqToSeqDataset(cfg, basic_data: BasicSequenceData, index_map, **kwargs):
    return ResNetLSTMSeqToSeqDataset(cfg, basic_data, index_map, **kwargs)


def partition_data(
    index_map,
    valid_samples,
    valid_all_samples,
    training_rate=0.9,
    valuation_rate=0.1,
    data_rate=1.0,
    shuffle=True,
    data_paths=None,
):

    if shuffle:
        np.random.shuffle(index_map)

    all_size = 0
    sum_valid_samples = valid_all_samples * data_rate

    accum_samples = 0.0
    for i in range(len(index_map)):
        accum_samples += valid_samples[index_map[i][0][0]]
        all_size = i
        if accum_samples > sum_valid_samples:
            break

    valuation_samples = sum_valid_samples * valuation_rate

    train_index_map, valuation_index_map = [], []
    accum_valuation_samples = 0

    # Track which trajectories land in each split for reproducibility logging
    val_traj, train_traj = [], []
    get_name = (
        (lambda idx: data_paths[idx] if data_paths and idx < len(data_paths) else f"traj_{idx}")
    )

    for i in range(all_size):
        traj_id = index_map[i][0][0]
        traj_name = get_name(traj_id)
        traj_samples = valid_samples[traj_id]

        if accum_valuation_samples < valuation_samples:
            valuation_index_map += index_map[i]
            accum_valuation_samples += traj_samples
            val_traj.append((traj_id, traj_name, traj_samples))
        else:
            train_index_map += index_map[i]
            train_traj.append((traj_id, traj_name, traj_samples))

    dropped_traj = []
    for i in range(all_size, len(index_map)):
        traj_id = index_map[i][0][0]
        traj_name = get_name(traj_id)
        traj_samples = valid_samples[traj_id]
        dropped_traj.append((traj_id, traj_name, traj_samples))

    logging.info(
        "[DATA_SPLIT] random_partition summary | data_rate=%.2f | valuation_rate=%.2f | "
        "selected_trajectories=%d | val=%d | train=%d | dropped=%d",
        data_rate,
        valuation_rate,
        all_size,
        len(val_traj),
        len(train_traj),
        len(dropped_traj),
    )
    logging.info(
        "[DATA_SPLIT] val trajectories: %s",
        [f"{t[0]}:{t[1]}({t[2]} samples)" for t in val_traj],
    )
    logging.info(
        "[DATA_SPLIT] train trajectories: %s",
        [f"{t[0]}:{t[1]}({t[2]} samples)" for t in train_traj],
    )
    if dropped_traj:
        logging.info(
            "[DATA_SPLIT] dropped trajectories (excluded by data_rate): %s",
            [f"{t[0]}:{t[1]}({t[2]} samples)" for t in dropped_traj],
        )

    return train_index_map, valuation_index_map
