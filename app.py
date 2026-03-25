"""
TartanIMU Gradio Demo
=====================
Interactive demo for neural inertial tracking.
Upload an .npz file with IMU data and get trajectory predictions + metrics.
"""

import argparse
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.configer import load_config, build_model, update_recursive
from dataloader import dataset_AirLab as dataset_utils
from evaluation import postprocess
from evaluation.metrics import compute_accruacy_metrics, compute_ate_rte
from model import function
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model / config – loaded once at startup
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None
CFG = None
ARGS = None


class _DummyArgs:
    def __init__(self):
        self.local_rank = 0
        self.world_size = 1
        self.log = False
        self.pdb = False
        self.exp_name = "demo"
        self.yaml = ""
        self.config = ""
        self.checkpoint = ""
        self.resume_from = ""


def load_model_and_config(config_path: str, checkpoint_dir: str):
    global MODEL, CFG, ARGS

    CFG = load_config(config_path)
    with open(CFG["model"]["model_yaml"], "r") as f:
        model_cfg = yaml.load(f, Loader=yaml.Loader)
    update_recursive(CFG, model_cfg)

    CFG["train"]["use_multi_gpu"] = False

    ARGS = _DummyArgs()
    ARGS.yaml = config_path
    ARGS.local_rank = 0

    model = build_model(ARGS, CFG)

    ckpt_files = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or "0"),
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    ckpt_path = os.path.join(checkpoint_dir, ckpt_files[-1])
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    MODEL = model
    logger.info("Model loaded from %s  (device=%s)", ckpt_path, DEVICE)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _torch_to_numpy(t):
    return t.cpu().detach().numpy()


def run_inference(npz_path: str):
    """Run full test inference on a single .npz file and return results dict."""
    cfg = CFG
    model = MODEL

    test_basic = dataset_utils.BasicSequenceData(cfg, [npz_path], mode="test")
    test_dataset = dataset_utils.ResNetLSTMSeqToSeqDataset(
        cfg, test_basic, test_basic.get_merged_index_map(), mode="test"
    )
    batch_size = cfg.get("test", {}).get("batch_size", 512)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(DEVICE)
    start_cov = cfg["train"]["start_cov_epochs"]

    targets_all, preds_all, preds_cov_all, orien_all = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in test_loader:
            batch = [t.to(device) for t in batch]
            pred, pred_cov, targ, ori, loss = function.fun_test_forward(
                cfg, model, batch, start_cov, epoch=9999
            )
            targets_all.append(_torch_to_numpy(targ))
            orien_all.append(_torch_to_numpy(ori))
            preds_all.append(_torch_to_numpy(pred))
            preds_cov_all.append(_torch_to_numpy(pred_cov))

    inference_time = time.time() - t0

    net_attr_dict = {
        "targets": np.concatenate(targets_all, axis=0),
        "orien": np.concatenate(orien_all, axis=0),
        "preds": np.concatenate(preds_all, axis=0),
        "preds_cov": np.concatenate(preds_cov_all, axis=0),
        "losses": [],
    }

    traj = postprocess.pose_integrate(
        cfg, test_dataset, net_attr_dict, cfg["data"]["use_local_coord"]
    )

    return traj, net_attr_dict, inference_time


def compute_metrics(traj, cfg):
    imu_freq_int = int(cfg["data"]["imu_freq"] * 1)
    ate, t_rte, d_rte = compute_ate_rte(traj["pos_pred"], traj["pos_gt"], imu_freq_int)
    ATE, AVE, P_RMSE, V_RMSE, X_ATE, Y_ATE, Z_ATE, X_AVE, Y_AVE, Z_AVE = (
        compute_accruacy_metrics(traj, imu_freq_int, cfg["data"]["use_local_coord"])
    )
    return {
        "ATE (m)": f"{ate:.4f}",
        "Translational RTE (m)": f"{t_rte:.4f}",
        "Drift RTE (m)": f"{d_rte:.4f}",
        "Position RMSE (m)": f"{P_RMSE:.4f}",
        "Velocity RMSE (m/s)": f"{V_RMSE:.4f}",
        "X / Y / Z ATE (m)": f"{X_ATE:.4f} / {Y_ATE:.4f} / {Z_ATE:.4f}",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_trajectory_figure(traj):
    pos_pred = traj["pos_pred"]
    pos_gt = traj["pos_gt"]
    ts = traj["ts"]

    fig = plt.figure(figsize=(18, 10), dpi=110)

    # 2D top-down (X-Y)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1], "r-", lw=2, label="Ground Truth")
    ax1.plot(pos_pred[:, 0], pos_pred[:, 1], "b--", lw=2, label="Predicted")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("2D Trajectory (X-Y)")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.3)

    # 2D side view (X-Z)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pos_gt[:, 0], pos_gt[:, 2], "r-", lw=2, label="Ground Truth")
    ax2.plot(pos_pred[:, 0], pos_pred[:, 2], "b--", lw=2, label="Predicted")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("2D Trajectory (X-Z)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3D trajectory
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], "r-", lw=2, label="Ground Truth")
    ax3.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], "b--", lw=2, label="Predicted")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_title("3D Trajectory")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Position error over time
    pos_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(ts - ts[0], pos_error, "g-", lw=1.5)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Position Error (m)")
    ax4.set_title("Position Error Over Time")
    ax4.grid(True, alpha=0.3)

    # Per-axis position comparison
    axis_labels = ["X", "Y", "Z"]
    colors = ["tab:red", "tab:green", "tab:blue"]
    ax5 = fig.add_subplot(2, 3, 5)
    for i in range(3):
        ax5.plot(ts - ts[0], pos_gt[:, i], "-", color=colors[i], lw=1, alpha=0.7, label=f"GT {axis_labels[i]}")
        ax5.plot(ts - ts[0], pos_pred[:, i], "--", color=colors[i], lw=1, alpha=0.7, label=f"Pred {axis_labels[i]}")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Position (m)")
    ax5.set_title("Per-Axis Position")
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # Cumulative distance
    ax6 = fig.add_subplot(2, 3, 6)
    cum_gt = np.cumsum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1))
    cum_pred = np.cumsum(np.linalg.norm(np.diff(pos_pred, axis=0), axis=1))
    ax6.plot(ts[1:] - ts[0], cum_gt, "r-", lw=2, label="Ground Truth")
    ax6.plot(ts[1:] - ts[0], cum_pred, "b--", lw=2, label="Predicted")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Cumulative Distance (m)")
    ax6.set_title("Cumulative Distance Traveled")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def make_imu_figure(npz_path: str):
    """Visualize the raw IMU data from the uploaded file."""
    data = np.load(npz_path)
    ts = data["retargetted_ts"]
    imu = data["retargetted_imu"]
    t_rel = ts - ts[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), dpi=110, sharex=True)

    for i, (lbl, c) in enumerate(zip(["X", "Y", "Z"], ["tab:red", "tab:green", "tab:blue"])):
        axes[0].plot(t_rel, imu[:, 3 + i], color=c, lw=0.8, label=lbl)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title("Accelerometer")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for i, (lbl, c) in enumerate(zip(["X", "Y", "Z"], ["tab:red", "tab:green", "tab:blue"])):
        axes[1].plot(t_rel, imu[:, i], color=c, lw=0.8, label=lbl)
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Gyroscope")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main Gradio callback
# ---------------------------------------------------------------------------

def predict(file):
    if file is None:
        raise gr.Error("Please upload an .npz file.")

    src_path = file if isinstance(file, str) else file.name

    if not src_path.endswith(".npz"):
        raise gr.Error("Only .npz files are supported.")

    # The dataloader infers motion type from the path; place the file under
    # a directory containing "human" so the correct head is selected.
    tmpdir = tempfile.mkdtemp()
    human_dir = os.path.join(tmpdir, "human")
    os.makedirs(human_dir, exist_ok=True)
    dst_path = os.path.join(human_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

    # Quick validation
    try:
        npz = np.load(dst_path)
        required = {"retargetted_ts", "retargetted_imu", "retargetted_pos", "retargetted_quat"}
        missing = required - set(npz.keys())
        if missing:
            raise gr.Error(f"NPZ file is missing required keys: {missing}")
        n_samples = len(npz["retargetted_ts"])
        duration = npz["retargetted_ts"][-1] - npz["retargetted_ts"][0]
        file_info = f"**Samples:** {n_samples}  |  **Duration:** {duration:.1f}s  |  **IMU shape:** {npz['retargetted_imu'].shape}"
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Failed to read .npz file: {e}")

    # IMU visualization
    imu_fig = make_imu_figure(dst_path)

    # Run inference
    try:
        traj, net_attr, inf_time = run_inference(dst_path)
    except Exception as e:
        logger.exception("Inference failed")
        raise gr.Error(f"Inference failed: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Metrics
    metrics = compute_metrics(traj, CFG)
    metrics["Inference Time (s)"] = f"{inf_time:.2f}"

    metrics_md = "| Metric | Value |\n|---|---|\n"
    for k, v in metrics.items():
        metrics_md += f"| {k} | {v} |\n"

    # Trajectory figure
    traj_fig = make_trajectory_figure(traj)

    # Downloadable results
    results_path = os.path.join(tempfile.gettempdir(), "tartanimu_results.npz")
    np.savez_compressed(
        results_path,
        ts=traj["ts"],
        pos_pred=traj["pos_pred"],
        pos_gt=traj["pos_gt"],
        cov_pred=traj["cov_pred"],
    )

    return file_info, imu_fig, traj_fig, metrics_md, results_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
#title { text-align: center; }
#metrics-box { font-size: 1.05em; }
"""

TITLE = "# TartanIMU – Neural Inertial Tracking Demo"
DESCRIPTION = """Upload an `.npz` file containing IMU data to run neural inertial tracking inference.

The file must contain: `retargetted_ts`, `retargetted_imu` (Nx6), `retargetted_pos` (Nx3), `retargetted_quat` (Nx4).

The model predicts a 3D trajectory from the IMU measurements and compares it against the ground-truth positions.
"""


def build_demo(example_files=None):
    with gr.Blocks(title="TartanIMU Demo") as demo:
        gr.Markdown(TITLE, elem_id="title")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(label="Upload .npz file", file_types=[".npz"])
                submit_btn = gr.Button("Run Inference", variant="primary", size="lg")
                file_info = gr.Markdown(label="File Info")
                metrics_output = gr.Markdown(label="Metrics", elem_id="metrics-box")
                result_file = gr.File(label="Download Results (.npz)")

            with gr.Column(scale=2):
                imu_plot = gr.Plot(label="Input IMU Data")
                traj_plot = gr.Plot(label="Trajectory Prediction")

        submit_btn.click(
            fn=predict,
            inputs=[input_file],
            outputs=[file_info, imu_plot, traj_plot, metrics_output, result_file],
        )

        if example_files:
            gr.Examples(
                examples=[[f] for f in example_files],
                inputs=[input_file],
                outputs=[file_info, imu_plot, traj_plot, metrics_output, result_file],
                fn=predict,
                cache_examples=False,
            )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TartanIMU Gradio Demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config/datasets/tartanimu/lamar.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing .pt checkpoints (default: inferred from config out_dir)",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    # Resolve checkpoint directory
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        tmp_cfg = load_config(args.config)
        ckpt_dir = os.path.join(tmp_cfg["train"]["out_dir"], "checkpoints")
    logger.info("Checkpoint dir: %s", ckpt_dir)

    load_model_and_config(args.config, ckpt_dir)

    # Gather example files if available – search common data locations
    example_files = []
    candidate_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data", "lamar-v2-NO-GRAVITY-split", "human", "test"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_data", "lamar-v2-processed-NO-GRAVITY", "human", "test"),
    ]
    for example_dir in candidate_dirs:
        if not os.path.isdir(example_dir):
            continue
        for root, _dirs, files in os.walk(example_dir):
            for f in sorted(files):
                if f.endswith(".npz"):
                    example_files.append(os.path.join(root, f))
                    if len(example_files) >= 3:
                        break
            if len(example_files) >= 3:
                break
        if example_files:
            break

    demo = build_demo(example_files if example_files else None)
    demo.queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
