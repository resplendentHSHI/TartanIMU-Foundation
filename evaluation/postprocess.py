import logging
import pdb
from os import path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def recover_global_pose_from_local_velocity(
    ts, velocity_local, initial_pos, initial_quat
):

    dt = ts[1:] - ts[:-1]
    dt = np.concatenate([dt, [dt[-1]]])
    n = len(dt)
    # Initialize global pose arrays
    pos_global = np.zeros((n, 3))
    # Set initial pose
    pos_global[0] = initial_pos

    current_rotation = Rotation.from_quat(initial_quat)

    print("n", n)
    for i in range(1, n):
        # Transform local velocity to global frame

        global_velocity = current_rotation[i - 1].apply(velocity_local[i - 1])

        # Integrate position
        pos_global[i] = pos_global[i - 1] + global_velocity * dt[i - 1]

    return pos_global


def align_trajectory_to_ground_truth(pred_positions, gt_positions):
    """
    Align predicted trajectory to ground truth trajectory.
    Ensures the predicted trajectory starts from the same initial position as ground truth.

    Args:
        pred_positions: Predicted positions [N, 3]
        gt_positions: Ground truth positions [N, 3]

    Returns:
        Aligned predicted positions [N, 3]
    """
    if len(pred_positions) != len(gt_positions):
        raise ValueError(
            f"Position arrays must have same length: {len(pred_positions)} vs {len(gt_positions)}"
        )

    # Get initial positions
    gt_initial = gt_positions[0]
    pred_initial = pred_positions[0]

    # Calculate the translation needed to align initial positions
    translation = gt_initial - pred_initial

    # Apply translation to align initial positions
    aligned_pred = pred_positions + translation

    return aligned_pred


def align_trajectory_with_scale_and_rotation(pred_positions, gt_positions):
    """
    Align predicted trajectory to ground truth using Umeyama algorithm.
    This provides more sophisticated alignment including scale and rotation.

    Args:
        pred_positions: Predicted positions [N, 3]
        gt_positions: Ground truth positions [N, 3]

    Returns:
        Aligned predicted positions [N, 3]
    """
    if len(pred_positions) != len(gt_positions):
        raise ValueError(
            f"Position arrays must have same length: {len(pred_positions)} vs {len(gt_positions)}"
        )

    # Center the trajectories
    gt_centered = gt_positions - np.mean(gt_positions, axis=0)
    pred_centered = pred_positions - np.mean(pred_positions, axis=0)

    # Compute covariance matrix
    H = pred_centered.T @ gt_centered

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute scale
    scale = np.sum(S) / np.sum(pred_centered**2)

    # Apply transformation
    aligned_pred_centered = scale * (pred_centered @ R.T)

    # Translate back to original coordinate system
    aligned_pred = aligned_pred_centered + np.mean(gt_positions, axis=0)

    return aligned_pred


def pose_integrate(cfg, dataset, preds_dict, use_local_coordinate=False):
    use_local_coordinate = cfg["data"]["use_local_coord"]
    preds = preds_dict["preds"]  # pose prediction
    preds_cov = preds_dict["preds_cov"]  # cov prediction
    window_time = cfg["model_param"]["window_time"]  # 1.0
    imu_freq = cfg["data"]["imu_freq"]  # 100.0 or 200.0
    seq_len = cfg["train"]["seq_len"]  # 10
    dp_t = window_time
    pred_vels = preds / dp_t  # 1204*3
    ind = np.array(
        [i[1] for i in dataset.index_map], dtype=np.int32
    )  # array([   0,    5,   10, ..., 6005, 6010, 6015])
    delta_int = int(window_time * imu_freq / 2.0)  # 50.0
    delta_int += int((seq_len - 1) * window_time * imu_freq)  # 50+900
    if not (window_time * imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = (
        ind + delta_int
    )  # array([  1900,   1905,   1910, ..., 123885, 123890, 123895])
    ts = dataset.ts[0]  
    dts = np.mean(
        ts[ind_intg[1:]] - ts[ind_intg[:-1]]
    )  # Mean of timestamps from index 955-6965 minus timestamps from 950-6960, average time difference between every 5 frames
    # Equivalent to obtaining the time difference between every 5 frames delta_t
    pos_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])  # 1205*3
    pos_intg1 = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])  # 1205*3
    gt_vel_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])  # 1205*3
    pos_intg[0] = dataset.gt_pos[0][
        ind_intg[0], 0 : pos_intg.shape[1]
    ]  # Take the GT position of frame 1900 of the current trajectory as the first frame
    # pos_intg1[0] = dataset.gt_pos[0][ind_intg[0], 0:pos_intg.shape[1]] #Take the GT position of frame 1900 of the current trajectory as the first frame
    gt_vel_intg[0] = dataset.gt_pos[0][ind_intg[0], 0 : pos_intg.shape[1]]
    preds_cov = np.concatenate([np.zeros([1, 3]), preds_cov])
    pos_gt = dataset.gt_pos[0][
        ind_intg[0] : ind_intg[-1], 0 : pos_intg.shape[1]
    ]  # Ground truth pose data from frame 950 to 6965, 6015*3
    pos_gt_rectify = pos_gt[::5]
    integrate_imu_frame = int(seq_len * imu_freq * 6 / (5))  # Integrate every 10s
    # dataset.gt_pos[0] is a list with length 7016, the first pose read is the ground truth pose at timestamp 950
    if use_local_coordinate:
        gt_ori = preds_dict["orien"][
            :, 0, :
        ]  # Take the rotation at the middle position of each 200-frame window [num_seq, 4]
        ## Transform local velocity to global frame
        current_rotation = Rotation.from_quat(gt_ori)
        pred_global_vel = current_rotation.apply(
            pred_vels
        )  # Apply rotation to predicted local velocity to convert to global coordinate system velocity #Global velocity of each 200-frame window, considered as the average velocity of frames 0-5 in that window
        gt_global_vel = current_rotation.apply(preds_dict["targets"])
        num_frames = pred_global_vel.shape[0]
        # for i in range(0, num_frames, integrate_imu_frame):
        #     end_idx = min(i + integrate_imu_frame, num_frames)
        #     pos_intg[i+1:end_idx+1] = np.cumsum(pred_global_vel[i:end_idx, :] * dts, axis=0) + pos_gt_rectify[i]
        pos_intg[1:] = np.cumsum(pred_global_vel[:, :] * dts, axis=0) + pos_intg[0]
        gt_vel_intg[1:] = np.cumsum(gt_global_vel[:, :] * dts, axis=0) + pos_intg[0]
    else:
        pos_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_intg[0]  # 1205*3
        pred_global_vel = pred_vels
        gt_global_vel = None

    ts_intg = np.append(
        ts[ind_intg], ts[ind_intg[-1]] + dts
    )  # Timestamp corresponding to frame 950 and frame 6965 plus additional 5-frame time interval
    ts_in_range = ts[
        ind_intg[0] : ind_intg[-1]
    ]  # In this time range, there are 6015 frames of data corresponding timestamps
    pos_pred = interp1d(ts_intg, pos_intg, axis=0)(
        ts_in_range
    )  # Interpolate predicted timestamps and pose information sampled at 20Hz into 6015 frames of data 6015*3
    if use_local_coordinate:
        gt_vel_in_pos = interp1d(ts_intg, gt_vel_intg, axis=0)(
            ts_in_range
        )  # Interpolate predicted timestamps and pose information sampled at 20Hz into 6015 frames of data 6015*3
        # Interpolate velocity predictions to match position timestamps
        vel_pred_interp = interp1d(ts[ind_intg], pred_global_vel, axis=0)(ts_in_range)
        vel_gt_interp = interp1d(ts[ind_intg], gt_global_vel, axis=0)(ts_in_range)
    else:
        # For non-local coordinate case, interpolate velocity predictions
        vel_pred_interp = interp1d(ts[ind_intg], pred_global_vel, axis=0)(ts_in_range)
        vel_gt_interp = None
    cov_pred = interp1d(ts_intg, preds_cov, axis=0)(ts_in_range)
    logging.info(
        f"pos_pred.shape:{pos_pred.shape} pos_gt.shape {pos_gt.shape} cov_pred.shape: {cov_pred.shape}"
    )

    # Debug: Check if initial positions are actually the same
    initial_diff = np.linalg.norm(pos_pred[0] - pos_gt[0])
    logging.info(f"Initial position difference: {initial_diff:.6f}")
    logging.info(f"GT initial position: {pos_gt[0]}")
    logging.info(f"Pred initial position: {pos_pred[0]}")

    traj_attr_dict = {
        "ts": ts_in_range,
        "pos_pred": pos_pred,  # Use original predicted positions
        "pos_gt": pos_gt,
        # "pos_gtvel_intg": gt_vel_in_pos,
        "cov_pred": cov_pred,
        "vel_pred": vel_pred_interp,  # Use interpolated velocity predictions
        "vel_gt": vel_gt_interp,
    }

    return traj_attr_dict


def compute_plot_dict(sample_freq, net_attr_dict, traj_attr_dict):
    ts = traj_attr_dict["ts"]  # 6015
    pos_pred = traj_attr_dict["pos_pred"]  # 6015*3 (already aligned)
    pos_gt = traj_attr_dict["pos_gt"]
    # pos_gtvel_intg = traj_attr_dict["pos_gtvel_intg"]
    # pred_local_vel = traj_attr_dict["pred_local_vel"]
    # gt_local_vel = traj_attr_dict["gt_local_vel"]

    total_pred = net_attr_dict["preds"].shape[0]  # 1204
    pred_ts = (1.0 / sample_freq) * np.arange(total_pred)  # (0-1203)/20.0
    pred_sigmas = np.exp(net_attr_dict["preds_cov"])  # (1204, 3)
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,  # Already aligned in pose_integrate
        "pos_gt": pos_gt,
        # "pos_gtvel_intg": pos_gtvel_intg,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "pred_sigmas": pred_sigmas,
    }

    return plot_dict


def plot_imus(feat, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    x = len(feat[:, 0])
    plt.subplot(2, 1, 1)
    for i in range(3):
        plt.plot(x, feat[:, i])
        plt.plot(x, feat[:, i])
    plt.ylabel("gyr")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t(s)")

    plt.subplot(2, 1, 2)
    for i in range(3):
        plt.plot(x, feat[:, 3 + i])
        plt.plot(x, feat[:, 3 + i])
    plt.ylabel("acc")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t(s)")
    return fig


def rotate_ax(step, ax):
    ax.view_init(30, step * 2)


def make_3d_trj_plots(plot_dict, outdir):
    pos_pred = plot_dict["pos_pred"]  # 6015*3
    pos_gt = plot_dict["pos_gt"]  # 6015*3
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], label="network_pred")
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label="Ground_truth")
    plt.legend(["network_pred", "Ground_truth"])
    if outdir is not None:
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(
            fig, rotate_ax, 180, fargs=(ax,), interval=100, blit=False
        )
        anim.save(osp.join(outdir, "3D_traj.gif"), writer="imagemagick", fps=20)


def make_plots(
    plot_dict, outdir, epoch_num, ave_ate, t_rte, d_rte, use_local=False, ratio=None
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    pos_pred = plot_dict["pos_pred"]  # 6015*3
    pos_gt = plot_dict["pos_gt"]  # 6015*3

    # Debug: Check initial positions in visualization
    initial_diff = np.linalg.norm(pos_pred[0] - pos_gt[0])
    print(f"[DEBUG] Visualization - Initial position difference: {initial_diff:.6f}")
    print(f"[DEBUG] GT initial: {pos_gt[0]}")
    print(f"[DEBUG] Pred initial: {pos_pred[0]}")
    print(f"[DEBUG] Are initial positions equal? {np.allclose(pos_pred[0], pos_gt[0])}")
    # if use_local:
    # pos_gtvel_intg = plot_dict["pos_gtvel_intg"]
    # pred_local_vel = plot_dict["pred_local_vel"]
    # gt_local_vel = plot_dict["gt_local_vel"]
    pred_ts = plot_dict["pred_ts"]  # 1204
    preds = plot_dict["preds"]  # 1204*3
    targets = plot_dict["targets"]  # 1204*3
    pred_sigmas = plot_dict["pred_sigmas"]  # 1204*3
    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1])  # Predicted trajectory, only plot x y 6015*3
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])  # Ground truth trajectory
    # if use_local:
    #     plt.plot(pos_gtvel_intg[:, 0], pos_gtvel_intg[:, 1])
    plt.axis("equal")
    if use_local:
        plt.legend(["network_pred", "Ground_truth", "GT_Vel_Intg"])
    else:
        plt.legend(["network_pred", "Ground_truth"])
    # plt.title(f"Epoch:{epoch_num} ATE:{ave_ate[-1]}, T_RTE:{t_rte[-1]}, D_RTE: {d_rte[-1]}")
    ratio_str = f"{ratio:.2%}" if ratio is not None else "N/A"
    # Add safety checks for empty lists
    ate_val = ave_ate[-1] if ave_ate else "N/A"
    t_rte_val = t_rte[-1] if t_rte else "N/A"
    d_rte_val = d_rte[-1] if d_rte else "N/A"
    plt.title(
        f"Epoch:{epoch_num}, Ratio:{ratio_str} ATE:{ate_val}, T_RTE:{t_rte_val}, D_RTE: {d_rte_val}"
    )
    plt.subplot2grid((3, 2), (2, 0), rowspan=1)
    # Add safety checks for empty lists before plotting
    if ave_ate:
        plt.plot(range(1, len(ave_ate) + 1), ave_ate, "-o", label="ATE", color="red")
    if t_rte:
        plt.plot(range(1, len(t_rte) + 1), t_rte, "-s", label="T_RTE", color="blue")
    if d_rte:
        plt.plot(
            range(1, len(d_rte) + 1), d_rte, "-^", label="Avg D_RTE", color="purple"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Metric Values")
    plt.title("Metrics over Epochs")
    plt.legend()
    plt.grid(True)
    # step = float(preds.shape[0] / pos_pred.shape[0])
    # for i in range(0, pos_pred.shape[0], 10):
    #     idx = int(np.around(i * step))
    #     pred = preds[idx]
    #     pred = 2 * pred / np.linalg.norm(pred)
    # plt.arrow(pos_gt[i, 0], pos_gt[i, 1], pred[0], pred[1], head_width=0.3)

    for i in range(
        preds.shape[1]
    ):  # 0 1 2 1204*3 preds contains the displacement distance difference of the last window of each sequence
        plt.subplot2grid((preds.shape[1], 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(
            targets[:, i]
        )  # Each frame of data is the distance difference of the last window in 10 windows of the 0th trajectory in ground truth (tail minus head)
        plt.legend(["network_pred", "Ground_truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)
    fig1.savefig(osp.join(outdir, f"2D_traj_{epoch_num}.png"))

    # fig2 = plt.figure(num="pred_sigma", dpi=dpi, figsize=figsize)
    # preds_plus_sig = preds + 3 * pred_sigmas
    # preds_minus_sig = preds - 3 * pred_sigmas
    # ylbs = ["x(m)", "y(m)", "z(m)"]
    # for i in range(preds.shape[1]):
    #     plt.subplot(preds.shape[1], 1, i + 1)
    #     plt.plot(pred_ts, preds_plus_sig[:, i], "-g", linewidth=0.2)
    #     plt.plot(pred_ts, preds_minus_sig[:, i], "-g", linewidth=0.2)
    #     plt.plot(pred_ts, preds[:, i], "-b", linewidth=0.5, label='pred')
    #     plt.plot(pred_ts, targets[:, i], "-r", linewidth=0.5, label='gt')
    #     plt.ylabel(ylbs[i])
    #     plt.legend()
    #     plt.grid(True)
    # plt.xlabel("t(s)")
    # fig2.savefig(osp.join(outdir, "pred_sigma.svg"))

    plt.close("all")


# def create_video_from_images(image_folder, output_video, fps=1):
#     # Get all files in the directory and sort them
#     images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
#     images.sort()

#     # Determine the width and height from the first image
#     frame = cv2.imread(os.path.join(image_folder, images[0]))
#     height, width, layers = frame.shape

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For an .mp4 output file
#     video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

#     for i, image in enumerate(images, start=1):
#         img_path = os.path.join(image_folder, image)
#         frame = cv2.imread(img_path)
#         # Put iteration number on each frame
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(frame, f'Iteration: {i}', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
#         video.write(frame)

#     cv2.destroyAllWindows()
#     video.release()
