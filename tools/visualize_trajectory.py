import logging
import os
import pdb
from os import path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def make_2Dtrj_plots(traject_txt, outdir):
    gt_all_data = np.loadtxt(traject_txt).astype(np.float32)
    if gt_all_data.shape[1] > 3:
        gt_ori = gt_all_data[:, 1:5]  # orientation.x->w GT
        gt_pos = gt_all_data[:, 5:]  # GT first 30 frames have unaligned IMU data position.x->z
    else:
        pos_gt = gt_pos
    # mpl.rcParams['legend.fontsize'] = 10
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(pos_gt[:, 0], pos_gt[:, 1], label='Ground_truth')
    # plt.legend(["trajectory"])
    dpi = 90
    figsize = (16, 9)
    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    # plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
    plt.axis("equal")
    plt.legend(["network_pred", "Ground_truth"])
    plt.title("2D trajectory")
    fig1.savefig(osp.join(outdir, "2Dtraj_intergration_pose.png"))
    plt.close("all")


def rotate_ax(step, ax):
    ax.view_init(30, step * 2)


def make_3Dtrj_plots(traject_txt, outdir):
    gt_all_data = np.loadtxt(traject_txt).astype(np.float32)
    if gt_all_data.shape[1] > 3:
        gt_ori = gt_all_data[:, 1:5]
        gt_pos = gt_all_data[:, 5:8]
    else:
        gt_pos = gt_all_data
    pos_gt = gt_pos
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    import pdb

    pdb.set_trace()
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label="Ground_truth")
    # for i in range(500):
    #     ax.plot(pos_gt[:i, 0], pos_gt[:i, 1], pos_gt[:i, 2], label='Ground_truth')
    plt.legend(["Ground_truth"])
    traj_name = traject_txt.split("/")[-1].split("_")[0]
    traj_num = traject_txt.split("/")[-2]
    out_path = osp.join(out_dir, traj_num)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if outdir is not None:
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(
            fig, rotate_ax, 180, fargs=(ax,), interval=100, blit=False
        )
        anim.save(
            osp.join(out_path, traj_name + "_3D_gt_pose.gif"),
            writer="imagemagick",
            fps=20,
        )


def make_accum_3Dtrj_plots(traject_txt, outdir):
    gt_all_data = np.loadtxt(traject_txt).astype(np.float32)
    if gt_all_data.shape[1] > 3:
        gt_ori = gt_all_data[:, 1:5]  # orientation.x->w GT
        gt_pos = gt_all_data[:, 5:8]  # GT first 30 frames have unaligned IMU data position.x->z
    else:
        gt_pos = gt_all_data  # GT first 30 frames have unaligned IMU data position.x->z
    pos_gt = gt_pos
    mpl.rcParams["legend.fontsize"] = 10
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    pos_gt = gt_data[::20]
    for i in tqdm(range(len(pos_gt.shape[0]))):
        ax.plot(pos_gt[:i, 0], pos_gt[:i, 1], pos_gt[:i, 2], label="Ground_truth")
    plt.legend(["Ground_truth"])
    traj_name = traject_txt.split("/")[-1].split("_")[0]
    traj_num = traject_txt.split("/")[-2]
    out_path = osp.join(out_dir, traj_num)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if outdir is not None:
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(
            fig, rotate_ax, 180, fargs=(ax,), interval=100, blit=False
        )
        anim.save(
            osp.join(out_path, traj_name + "_3D_gt_pose.gif"),
            writer="imagemagick",
            fps=20,
        )


def make_3Dtrj_plots_all(traj_path, outdir):
    files = os.listdir(traj_path + "/")
    for j in files:
        if "gt" in j:
            traject_txt = osp.join(traj_path, j)
            gt_all_data = np.loadtxt(traject_txt).astype(np.float32)
            if gt_all_data.shape[1] > 3:
                gt_ori = gt_all_data[:, 1:5]  # orientation.x->w GT
                gt_pos = gt_all_data[:, 5:8]  # GT first 30 frames have unaligned IMU data position.x->z
            else:
                gt_pos = gt_all_data  # GT first 30 frames have unaligned IMU data position.x->z
            pos_gt = gt_pos
            mpl.rcParams["legend.fontsize"] = 10
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], label="Ground_truth")
            plt.legend(["Ground_truth"])
            traj_name = j.split("_")[0]
            traj_num = traj_path.split("/")[-1]
            out_path = osp.join(out_dir, traj_num)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if outdir is not None:
                import matplotlib.animation as animation

                anim = animation.FuncAnimation(
                    fig, rotate_ax, 180, fargs=(ax,), interval=100, blit=False
                )
                anim.save(
                    osp.join(out_path, traj_name + "_3D_gt_pose.gif"),
                    writer="imagemagick",
                    fps=20,
                )


def calculate_velocity_from_poses(ts, pos, quat):
    pos_gt = pos - pos[0]  # Ground truth positions relative to the starting point
    quat_gt = Rotation.from_quat(quat)
    dt = ts[1:] - ts[:-1]

    velocity_global = (pos_gt[1:] - pos_gt[:-1]) / dt[:, np.newaxis]

    velocity_body = quat_gt[:-1].inv().apply(velocity_global)

    return velocity_global, velocity_body


def plot_data(
    pos_gt,
    pos_local,
    gt_pos_dis,
    local_pos_dis,
    gt_ori,
    velocity_body,
    velocity_global,
    sample_vel_global,
    sample_vel_body,
):
    # Plot position
    import matplotlib.gridspec as gridspec

    plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(9, 1)  # 7 rows, 1 column
    plt.subplot(gs[0, 0])
    plt.plot(pos_gt[:, 0], label="X")
    plt.plot(pos_gt[:, 1], label="Y")
    plt.plot(pos_gt[:, 2], label="Z")
    plt.title("Position (pos_gt)")
    plt.legend()

    plt.subplot(gs[1, 0])
    plt.plot(pos_local[:, 0], label="x")
    plt.plot(pos_local[:, 1], label="y")
    plt.plot(pos_local[:, 2], label="z")
    plt.title("Body Position (pose_local)")
    plt.legend()

    plt.subplot(gs[2, 0])
    plt.plot(gt_pos_dis[:, 0], label="x")
    plt.plot(gt_pos_dis[:, 1], label="y")
    plt.plot(gt_pos_dis[:, 2], label="z")
    plt.title("global pos_displacement")
    plt.legend()

    plt.subplot(gs[3, 0])
    plt.plot(local_pos_dis[:, 0], label="x")
    plt.plot(local_pos_dis[:, 1], label="y")
    plt.plot(local_pos_dis[:, 2], label="z")
    plt.title("local pos_displacement")
    plt.legend()

    # Plot orientation (quaternion components)
    # plt.subplot(6, 1, 5)
    # plt.plot(gt_ori[:, 0], label='Qx')
    # plt.plot(gt_ori[:, 1], label='Qy')
    # plt.plot(gt_ori[:, 2], label='Qz')
    # plt.plot(gt_ori[:, 3], label='Qw')
    # plt.title('Orientation (gt_ori)')
    # plt.legend()

    # Plot body frame velocity
    plt.subplot(gs[4:6, 0])
    plt.plot(velocity_body[:, 0], label="Vx")
    plt.plot(velocity_body[:, 1], label="Vy")
    plt.plot(velocity_body[:, 2], label="Vz")
    plt.title("Body Frame Velocity (velocity_body)")
    plt.legend()

    plt.subplot(gs[7:8, 0])
    plt.plot(velocity_global[:, 0], label="Vx")
    plt.plot(velocity_global[:, 1], label="Vy")
    plt.plot(velocity_global[:, 2], label="Vz")
    plt.title("Global Frame Velocity (velocity_global)")
    plt.legend()

    # plt.subplot(gs[7, 0])
    # plt.plot(sample_vel_body[:, 0], label='Vx')
    # plt.plot(sample_vel_body[:, 1], label='Vy')
    # plt.plot(sample_vel_body[:, 2], label='Vz')
    # plt.title('sampled Body Frame Velocity (velocity_body)')
    # plt.legend()

    # plt.subplot(gs[8, 0])
    # plt.plot(sample_vel_global[:, 0], label='Vx')
    # plt.plot(sample_vel_global[:, 1], label='Vy')
    # plt.plot(sample_vel_global[:, 2], label='Vz')
    # plt.title('sampled Global Frame Velocity (velocity_Global)')
    # plt.legend()

    plt.tight_layout()
    plt.savefig("tools/data_plot_new.png")


def recover_global_pose_from_local_velocity(
    ts, velocity_local, initial_pos, initial_quat
):
    n = len(ts)
    dt = ts[1:] - ts[:-1]

    # Initialize global pose arrays
    pos_global = np.zeros((n, 3))
    # Set initial pose
    pos_global[0] = initial_pos

    current_rotation = Rotation.from_quat(initial_quat)

    for i in range(1, n):
        # Transform local velocity to global frame
        global_velocity = current_rotation[i - 1].apply(velocity_local[i - 1])

        # Integrate position
        pos_global[i] = pos_global[i - 1] + global_velocity * dt[i - 1]

    return pos_global


def load(data_path, verbose=False, use_local_coord=False):

    # read data from npz file
    all_data = np.load(data_path)
    ts = all_data["retargetted_ts"]
    imu = all_data["retargetted_imu"]
    pos = all_data["retargetted_pos"]
    quat = all_data["retargetted_quat"]  # in xyzw format
    accel_body = imu[:, :3]
    angular_vel_body = imu[:, 3:]

    velocity_global, velocity_body = calculate_velocity_from_poses(ts, pos, quat)
    # for i in range(1899, velocity_global.shape[0], 5):
    #     vel_mean = np.mean(velocity_global[i:i+5], axis=0, keepdims=True)
    #     vel_body_mean = np.mean(velocity_body[i:i+5], axis=0, keepdims=True)

    # TODO: Only for Tartan Drive Dataset
    pos_gt = pos - pos[0]
    quat_gt = Rotation.from_quat(quat)

    # transform body frame imu measurements into global coordinate for easier learning
    # this requires rotation from body to world frame. There are two options to get it:
    #   1. use ground truth rotation from vio supervision signal
    #   2. (pre)integrate imu data and obtain a approximate

    gravity = 9.8101
    global_to_local_rotation = None

    if not use_local_coord:
        # case 1: use ground truth rotation from vio supervision signal
        global_to_local_rotation = quat_gt
        accel_est = global_to_local_rotation.apply(accel_body)
        accel_est -= np.array([0, 0, gravity])
        angular_vel_est = global_to_local_rotation.apply(angular_vel_body)

    else:
        # case 2: (pre)integrate imu data and obtain a approximate TODO: Need to fix the parameter names
        local_to_global_rotation = quat_gt.inv()
        accel_est = accel_body
        angular_vel_est = angular_vel_body

    print(pos_gt.shape, quat_gt.as_quat().shape, velocity_body.shape)
    gt_disp = pos_gt[200:] - pos_gt[:-200]  # Get the ground truth position offset corresponding to each window_size

    seq_len = ts.shape[0]
    imu_freq = 200
    if seq_len < imu_freq * 10:
        return False

    dt = ts[1] - ts[0]  # ts already clean

    get_gt = False
    data_valid = True
    sum_duration = ts[-1] - ts[0]

    if verbose:
        logging.info(f"{data_path}: sum time: {sum_duration}, gt: {get_gt}")

    new_ts = ts[:, np.newaxis]
    # import pdb; pdb.set_trace()

    ts = new_ts
    # self.features = np.concatenate([glob_gyro, glob_acce, tmp_gyro, tmp_acce], axis=1)
    features = np.concatenate(
        [angular_vel_est, accel_est], axis=1
    )  # global gyro and accel
    orientations = quat_gt.as_quat()  # [x, y, z, w] Ground truth rotation of current frame in world coordinate system
    pos_gt = pos_gt  # Ground truth translation of current frame in world coordinate system
    gt_ori = (
        quat_gt.as_quat()
    )  # Ground truth rotation of current frame in world coordinate system, consistent with the orientation values above

    features = features[:-1]
    orientations = orientations[:-1]
    pos_gt = pos_gt[:-1]
    gt_ori = gt_ori[:-1]
    targets = gt_disp[:-1]
    print(ts.shape)
    print(features.shape, gt_ori.shape, velocity_body.shape)
    assert (
        features.shape[0]
        == pos_gt.shape[0]
        == gt_ori.shape[0]
        == velocity_body.shape[0]
    )

    pos_recover = recover_global_pose_from_local_velocity(
        ts, velocity_body, pos_gt[0], gt_ori
    )

    quat_gt = Rotation.from_quat(gt_ori)
    local_disp = quat_gt[200:].inv().apply(pos_gt[200:]) - quat_gt[:-200].inv().apply(
        pos_gt[:-200]
    )  # Get the ground truth position offset corresponding to each window_size
    pdb.set_trace()
    print(local_disp.shape, targets.shape)

    plot_data(
        pos_gt,
        pos_recover,
        targets,
        local_disp,
        gt_ori,
        velocity_body,
        velocity_global,
        sample_vel_global=None,
        sample_vel_body=None,
    )


    return True


if __name__ == "__main__":
    # traj_txt = '/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/sifan/slam-package/rnin-vio/dataset/TRO_datatset/processed_data/test/traj1/gt_200HZ.txt'
    # out_dir='/mnt/beegfs/ssd_pool/docker/user/hadoop-automl/sifan/slam-package/rnin-vio/dataset/TRO_datatset/save2/'
    # make_3Dtrj_plots(traj_txt, out_dir)
    npz_path = "/ocean/projects/cis220039p/shiboz/dataset_process/data/9_Motion_Clean/debug_dataset/car/train/112/1.npz"
    load(npz_path, verbose=True, use_local_coord=False)
