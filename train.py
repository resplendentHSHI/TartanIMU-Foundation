"""
This is for neural inertial tracking.
"""

# from torch.utils.tensorboard import SummaryWriter
import logging
import os
import pdb
import time
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from model import function

# from utils import log_setting  # Not available
from tools import distributed_eval
from tqdm import tqdm

# Try to import lora (may not be available in all environments)
try:
    import lora

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    logging.warning("LoRA not available - online adaptation may not work properly")

from model.model_factory import ModelFactory
from utils.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_PATIENCE,
    MOTION_TYPES,
    torch_to_numpy,
)
from utils.error_handling import TrainingError, handle_exceptions
from utils.logging_config import get_logger, log_to_wandb
from utils.rich_logging import (
    alert,
    attention_grabber,
    banner,
    celebration,
    critical,
    debug,
    error,
    flash_message,
    highlight,
    info,
    info_highlight,
    print_training_stats,
    rich_logger,
    success,
    success_highlight,
    warning,
    warning_highlight,
)

logger = get_logger(__name__)


def write_wandb(header, objs, epoch_i, local_rank=0):
    """Write metrics to wandb."""
    # Only log from rank 0 to avoid duplicate entries
    if wandb.run is not None and local_rank == 0:
        wandb.log({header: objs}, step=epoch_i)


from model.model_factory import ModelFactory

# Utility functions moved to appropriate modules
from utils.constants import torch_to_numpy
from utils.logging_config import log_to_wandb


def log_training_metrics(summary_writer, mode, ml_loss, epoch, optimizer):
    """Log training metrics to tensorboard."""
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(f"{mode}: average ml loss: {ml_loss}")


def save_model(
    path, epoch, network, optimizer, use_multi_gpu=False, local_rank=0, save_reason=""
):
    """Save model checkpoint with enhanced error handling, multiple checkpoint support, and rotation."""
    try:
        # Validate output directory
        if not os.path.isdir(path):
            logging.error(f"Invalid output directory: {path}")
            raise ValueError(f"Invalid output directory: {path}")

        # Create checkpoints directory
        checkpoints_dir = osp.join(path, "checkpoints")
        if not osp.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir, exist_ok=True)
            logging.info(f"Created checkpoints directory: {checkpoints_dir}")

        # Always save epoch-specific checkpoint for history
        epoch_checkpoint = osp.join(checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")

        # Additional best model checkpoints (these get overwritten)
        best_checkpoints = []
        if save_reason == "improved_val_loss":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_val_loss.pt")
            )
        elif save_reason == "improved_train_loss":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_train_loss.pt")
            )
        elif save_reason == "validation_better_than_train":
            best_checkpoints.append(
                osp.join(checkpoints_dir, "checkpoint_best_val_loss.pt")
            )

        # Use epoch checkpoint as primary path
        model_path = epoch_checkpoint

        # Prepare state dict (handle both wrapped and unwrapped models)
        if hasattr(network, "module"):
            # Model is wrapped in DistributedDataParallel
            state_dict = {
                "model_state_dict": network.module.state_dict() if hasattr(network, "module") else network.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
            }
        else:
            # Model is not wrapped
            state_dict = {
                "model_state_dict": network.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
            }

        # Save epoch-specific checkpoint
        torch.save(state_dict, model_path)
        logging.info(f"Epoch checkpoint saved to {model_path}")

        # Save best model checkpoints (if this is a best model)
        for best_path in best_checkpoints:
            torch.save(state_dict, best_path)
            logging.info(f"Best model checkpoint saved to {best_path}")

        # Implement checkpoint rotation (keep last 10 epochs + best models)
        cleanup_old_checkpoints(checkpoints_dir, keep_last=10)

        # Verify files were created
        all_paths = [model_path] + best_checkpoints
        for path_to_check in all_paths:
            if os.path.exists(path_to_check):
                file_size = os.path.getsize(path_to_check)
                logging.info(
                    f"Checkpoint verified. Size: {file_size} bytes - {path_to_check}"
                )
            else:
                logging.error(f"Checkpoint file was not created: {path_to_check}")

    except Exception as e:
        logging.error(f"Error saving model checkpoint: {e}")
        logging.error(
            f"Path: {path}, Epoch: {epoch}, Use multi-GPU: {use_multi_gpu}, Rank: {local_rank}"
        )
        raise


def cleanup_old_checkpoints(checkpoints_dir, keep_last=10):
    """Clean up old epoch checkpoints, keeping only the most recent ones and best models."""
    try:
        import glob
        import re

        # Find all epoch checkpoints
        epoch_pattern = osp.join(checkpoints_dir, "checkpoint_epoch_*.pt")
        epoch_files = glob.glob(epoch_pattern)

        if len(epoch_files) <= keep_last:
            return  # No cleanup needed

        # Extract epoch numbers and sort
        epoch_info = []
        for filepath in epoch_files:
            filename = osp.basename(filepath)
            match = re.search(r"checkpoint_epoch_(\d+)\.pt", filename)
            if match:
                epoch_num = int(match.group(1))
                epoch_info.append((epoch_num, filepath))

        # Sort by epoch number and keep only the most recent
        epoch_info.sort(key=lambda x: x[0])
        files_to_remove = epoch_info[:-keep_last]  # Remove all but last N

        for epoch_num, filepath in files_to_remove:
            try:
                os.remove(filepath)
                logging.info(f"Removed old checkpoint: {filepath}")
            except OSError as e:
                logging.warning(f"Could not remove old checkpoint {filepath}: {e}")

    except Exception as e:
        logging.warning(f"Error during checkpoint cleanup: {e}")


def freeze_backbone_parameters(model):
    """Freeze backbone parameters for finetuning."""
    for name, param in model.named_parameters():
        # Freeze backbone parameters (typically the feature extraction layers)
        # This is a simple implementation - you may need to adjust based on your model structure
        if "backbone" in name or "trunk" in name or "encoder" in name:
            param.requires_grad = False
            logging.info(f"Frozen parameter: {name}")
        else:
            param.requires_grad = True
            logging.info(f"Trainable parameter: {name}")

    return model  # Return the modified model


class Trainer:
    """Simplified trainer class for neural inertial tracking."""

    def __init__(self, args, cfg, model, optimizer, start_epoch=0):
        # Basic setup
        self.local_rank = args.local_rank
        self.cfg = cfg
        self.device = torch.device(
            f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.optimizer = optimizer
        self.start_epoch = start_epoch

        # Configuration
        self.use_multi_gpu = cfg["train"]["use_multi_gpu"]
        self.use_amp = cfg["train"]["use_amp"]
        self.epochs = cfg["train"]["epochs"]
        self.start_cov_epochs = cfg["train"]["start_cov_epochs"]
        self.out_dir = cfg["train"]["out_dir"]

        # Debug output directory
        logging.info(f"Trainer initialization - Output directory: {self.out_dir}")
        logging.info(f"Trainer initialization - Local rank: {self.local_rank}")
        logging.info(f"Trainer initialization - Use multi-GPU: {self.use_multi_gpu}")

        # Validate output directory
        if not os.path.exists(self.out_dir):
            logging.warning(f"Output directory does not exist: {self.out_dir}")
            try:
                os.makedirs(self.out_dir, exist_ok=True)
                logging.info(f"Created output directory: {self.out_dir}")
            except Exception as e:
                logging.error(f"Failed to create output directory: {e}")
        elif not os.path.isdir(self.out_dir):
            logging.error(f"Output path exists but is not a directory: {self.out_dir}")
        else:
            logging.info(f"Output directory is valid: {self.out_dir}")

        # Setup scheduler
        self._setup_scheduler()

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Loss tracking for checkpointing
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.last_save_epoch = 0
        self.significant_improvement_threshold = 0.005  # 0.5% improvement threshold

        # Logging setup
        self.log = args.log
        logger.info(f"Trainer initialized on device: {self.device}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        try:
            from utils.improved_scheduler import create_improved_scheduler

            self.scheduler = create_improved_scheduler(
                self.optimizer, self.cfg, scheduler_type="multihead_plateau"
            )
            logger.info("Using improved multi-head scheduler")
        except ImportError:
            # Fallback to standard scheduler
            scheduler_cfg = self.cfg["train"]["scheduler"]
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=scheduler_cfg["factor"],
                patience=scheduler_cfg["patience"],
                eps=1e-6,
            )
            logger.warning("Using standard ReduceLROnPlateau scheduler")

    def inference_step(self, data_loader, epoch):
        """Perform inference on validation/test data."""
        self.model.eval()
        all_results = {"targets": [], "preds": [], "preds_cov": [], "losses": []}

        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                pred, pred_cov, target, orien, loss = function.fun_test_forward(
                    self.cfg, self.model, batch, self.start_cov_epochs, epoch
                )

                # Collect results
                all_results["targets"].append(target)
                all_results["preds"].append(pred)
                all_results["preds_cov"].append(pred_cov)
                all_results["losses"].append(loss.unsqueeze(0))

        # Concatenate results
        results = self._concatenate_results(all_results, len(data_loader.dataset))

        # Convert to numpy for evaluation
        return {key: torch_to_numpy(value) for key, value in results.items()}

    def _move_batch_to_device(self, batch):
        """Move batch tensors to appropriate device."""
        if self.use_multi_gpu:
            return [t.cuda(self.local_rank, non_blocking=True) for t in batch]
        else:
            return [t.to(self.device) for t in batch]

    def _concatenate_results(self, all_results, dataset_size):
        """Concatenate results from all batches."""
        concatenated = {}
        for key, values in all_results.items():
            concatenated_tensor = torch.concat(values, dim=0)
            if self.use_multi_gpu:
                concatenated[key] = distributed_eval.distributed_concat(
                    concatenated_tensor, dataset_size
                )
            else:
                concatenated[key] = concatenated_tensor
        return concatenated

    def train_step(self, data_loader, epoch, fix_backbone=False, current_frame=None):
        train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
        self.model.train()
        # Will freeze the backbone and train multi-head
        # if fix_backbone:
        #     self.model=freeze_backbone_parameters(self.model)
        # test trainable params - only print once per training session
        if not hasattr(self, "_params_printed"):
            ModelFactory.print_trainable_parameters(self.model)
            self._params_printed = True
        torch.cuda.synchronize()
        data_start = time.time()
        iteration = 0
        if current_frame is not None:
            logging.info(
                f"-------------- Training, current trajectory length time {int(current_frame/200)} s---------------"
            )
        else:
            logging.info("-------------- Training ---------------")
        for bid, batch in enumerate(data_loader):
            iteration = iteration + 1
            if self.use_multi_gpu:
                batch = [t.cuda(self.local_rank, non_blocking=True) for t in batch]
            else:
                batch = [t.to(self.device) for t in batch]
            torch.cuda.synchronize()
            data_end = time.time()
            data_time = data_end - data_start  # data load time
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred, pred_cov, targ, loss = function.fun_train_forward_efficient(
                    self.cfg, self.model, batch, self.start_cov_epochs, epoch
                )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # Only log train loss every 1000 steps to reduce output frequency
            if hasattr(self, "train_log_counter"):
                self.train_log_counter += 1
            else:
                self.train_log_counter = 0

            if self.train_log_counter % 1000 == 0:
                logging.info(f"train loss: {loss}")
            torch.cuda.synchronize()
            back_end = time.time()
            inferback_time = back_end - data_end  # training and backward time
            # Log timing every 100 iterations to diagnose bottlenecks
            if iteration % 100 == 0:
                logging.info(f"Batch {iteration}: data_time={data_time:.4f}s, compute_time={inferback_time:.4f}s, ratio={data_time/inferback_time:.2f}")
            # pbar.set_description(f'Progress {iteration}% Fix Backbone {fix_backbone}')
            # pbar.set_postfix(inferback_time=inferback_time, data_loader_time=data_time, refresh=False)
            # pbar.update(1)
            data_start = time.time()  # Reset for next iteration
            # Keep tensors on GPU during training loop to avoid blocking GPU pipeline
            # Convert to numpy only at the end to improve GPU utilization
            train_targets.append(targ.detach())  # [10*3]
            train_preds.append(pred.detach())  # [10*3]
            train_preds_cov.append(pred_cov.detach())  # [10*3]
            train_losses.append(loss.detach())

        # print('bn weight in resnet_post_pro',self.model.module.model.resnet_post_pro[1].weight[0] if hasattr(self.model, "module") else self.model.model.resnet_post_pro[1].bias[0])
        # print('bn bias in resnet_post_pro',self.model.module.model.resnet_post_pro[1].bias[0] if hasattr(self.model, "module") else self.model.model.resnet_post_pro[1].bias[0])
        # Convert all tensors to numpy at once after training loop completes
        # This avoids blocking GPU pipeline during training
        train_targets = np.concatenate([torch_to_numpy(t) for t in train_targets], axis=0)
        train_preds = np.concatenate([torch_to_numpy(t) for t in train_preds], axis=0)
        train_preds_cov = np.concatenate([torch_to_numpy(t) for t in train_preds_cov], axis=0)
        train_losses = [torch_to_numpy(l) for l in train_losses]
        train_attr_dict = {
            "targets": train_targets,
            "preds": train_preds,
            "preds_cov": train_preds_cov,
            "losses": train_losses,
        }
        return train_attr_dict

    def train(self, train_loader, val_loader=None, test_loader=None):
        # Use existing tracking variables if resuming, otherwise initialize
        if self.start_epoch == 0:
            # Fresh training - initialize tracking variables
            self.best_val_loss = float("inf")
            self.best_train_loss = float("inf")
            self.last_save_epoch = 0
            logging.info("Initialized tracking variables for fresh training")
        else:
            # Resuming training - keep existing tracking variables
            logging.info(
                f"Resuming training from epoch {self.start_epoch} with existing tracking variables"
            )

        epoch_val_loss, epoch_val_mse, epoch_test_loss = [], [], []
        epoch_train_loss, epoch_train_mse, epoch_test_mse = [], [], []

        # main training loop
        for epoch in range(self.start_epoch + 1, self.epochs):
            rich_logger.training_header(epoch, self.epochs, "Training")
            validation_loss, validation_mse, validation_time, test_loss_val = (
                [],
                [],
                [],
                [],
            )
            test_mse_val, test_time = [], []

            torch.cuda.synchronize()
            start_t = time.time()
            # if (epoch/self.epochs) > 0.7:
            # print('freeze the backbone')
            train_attr_dict = self.train_step(train_loader, epoch)
            # else:
            #     print('not freeze the backbone')
            #     train_attr_dict = self.train_step(train_loader, epoch) #the dictionary contains four keys: ['targets', 'preds', 'preds_cov', 'losses']

            # write_summary(self.summary_writer, train_attr_dict, epoch, self.optimizer, "train")
            torch.cuda.synchronize()
            end_t = time.time()
            epoch_time = end_t - start_t
            info(f"Whole epoch time: {epoch_time:.2f}s")
            train_loss = np.average(train_attr_dict["losses"])
            train_mse_val = np.mean(
                (train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2
            )
            instance_per_second = len(train_loader.dataset) / epoch_time

            if self.log:
                write_wandb("train", train_loss, epoch, self.local_rank)
                write_wandb("run_time", epoch_time, epoch, self.local_rank)
                write_wandb(
                    "instance_per_second", instance_per_second, epoch, self.local_rank
                )
                write_wandb("train_mse_val", train_mse_val, epoch, self.local_rank)
                write_wandb(
                    "lr", self.optimizer.param_groups[0]["lr"], epoch, self.local_rank
                )

            epoch_train_loss.append(train_loss)  # The loss for each epoch is the mean of all batch losses
            epoch_train_mse.append(train_mse_val)

            # Run validation if available
            validation_loss = None
            validation_mse = None
            if val_loader is not None:
                start_t = time.time()
                val_attr_dict = self.inference_step(val_loader, epoch)
                end_t = time.time()

                validation_loss = np.average(val_attr_dict["losses"])
                validation_mse = np.mean(
                    (val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2
                )
                validation_time = end_t - start_t

                if self.log:
                    write_wandb(
                        "validation_loss", validation_loss, epoch, self.local_rank
                    )
                    write_wandb(
                        "validation_mse", validation_mse, epoch, self.local_rank
                    )
                    write_wandb(
                        "validation_time", validation_time, epoch, self.local_rank
                    )

                logging.info(f"val loss: {validation_loss:.6f}, val mse: {validation_mse:.6f}")
                epoch_val_loss.append(validation_loss)
                epoch_val_mse.append(validation_mse)

            # Unified scheduler step (always use training loss)
            current_train_loss = np.average(train_attr_dict["losses"])
            if hasattr(self.scheduler, "step") and callable(
                getattr(self.scheduler, "step", None)
            ):
                self.scheduler.step(current_train_loss)
            logging.info(
                f"LR scheduler stepped with training loss: {current_train_loss:.6f}"
            )

            # Run test set if available
            test_loss_val = None
            if test_loader is not None:
                start_t = time.time()
                test_attr_dict = self.inference_step(test_loader, epoch)
                end_t = time.time()
                test_loss_val = np.average(test_attr_dict["losses"])
                test_mse_val = np.mean(
                    (test_attr_dict["targets"] - test_attr_dict["preds"]) ** 2
                )
                test_time = end_t - start_t

                if self.log:
                    write_wandb("test_loss", test_loss_val, epoch, self.local_rank)
                    write_wandb("test_mse", test_mse_val, epoch, self.local_rank)
                    write_wandb("test_time", test_time, epoch, self.local_rank)

                epoch_test_loss.append(test_loss_val)
                epoch_test_mse.append(test_mse_val)

                # Display test statistics
                print_training_stats(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=validation_loss if validation_loss is not None else None,
                    test_loss=test_loss_val,
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch_time=epoch_time,
                )

                if self.optimizer.param_groups[0]["lr"] < 1.1e-6:
                    break

            # Unified checkpoint saving strategy
            self.save_checkpoint_strategy(
                epoch, current_train_loss, validation_loss, test_loss_val
            )

        # Only finish wandb from rank 0
        if self.local_rank == 0:
            wandb.finish()

    def _plot_epoch_loss_results(
        self,
        test_loader,
        val_loader,
        epoch_train_loss,
        epoch_train_mse,
        epoch_val_loss,
        epoch_val_mse,
        epoch_test_loss,
        epoch_test_mse,
    ):
        ## save and plot epoch-loss with enhanced styling and metrics
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Create figure with enhanced styling
        fig = plt.figure(num="Training Loss Analysis", dpi=150, figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

        # Main loss plot
        ax_main = fig.add_subplot(gs[0, 0])

        # Prepare data
        train_loss = np.array(epoch_train_loss)
        train_mse = np.array(epoch_train_mse)
        epochs = range(1, len(train_loss) + 1)

        # Enhanced plotting with better colors and styling
        colors = {
            "train": "#1f77b4",  # Blue
            "val": "#d62728",  # Red
            "test": "#2ca02c",  # Green
        }

        # Plot training loss
        line_train = ax_main.plot(
            epochs,
            train_loss,
            color=colors["train"],
            linewidth=2.5,
            label="Training Loss",
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

        loss_all = train_loss[:, np.newaxis]
        mse_all = train_mse[:, np.newaxis]

        # Plot validation loss if available
        if val_loader is not None:
            val_loss = np.array(epoch_val_loss)
            val_mse = np.array(epoch_val_mse)
            line_val = ax_main.plot(
                epochs,
                val_loss,
                color=colors["val"],
                linewidth=2.5,
                label="Validation Loss",
                marker="s",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            loss_all = np.concatenate((loss_all, val_loss[:, np.newaxis]), axis=1)
            mse_all = np.concatenate((mse_all, val_mse[:, np.newaxis]), axis=1)

        # Plot test loss if available
        if test_loader is not None:
            test_loss = np.array(epoch_test_loss)
            test_mse = np.array(epoch_test_mse)
            line_test = ax_main.plot(
                epochs,
                test_loss,
                color=colors["test"],
                linewidth=2.5,
                label="Test Loss",
                marker="^",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            loss_all = np.concatenate((loss_all, test_loss[:, np.newaxis]), axis=1)
            mse_all = np.concatenate((mse_all, test_mse[:, np.newaxis]), axis=1)

        # Enhanced styling for main plot
        ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
        ax_main.set_title(
            "Training Progress: Loss vs Epochs", fontsize=16, fontweight="bold", pad=20
        )
        ax_main.grid(True, alpha=0.3, linestyle="--")
        ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

        # Add trend analysis
        if len(train_loss) > 1:
            # Calculate trend
            trend = np.polyfit(epochs, train_loss, 1)
            trend_line = np.poly1d(trend)(epochs)
            ax_main.plot(
                epochs,
                trend_line,
                "--",
                color="gray",
                alpha=0.7,
                linewidth=1.5,
                label="Trend",
            )

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_metrics.axis("off")

        # Calculate important metrics
        final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
        min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
        max_train_loss = np.max(train_loss) if len(train_loss) > 0 else 0
        loss_improvement = (
            ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
            if len(train_loss) > 0 and train_loss[0] > 0
            else 0
        )

        if val_loader is not None:
            final_val_loss = val_loss[-1] if len(val_loss) > 0 else 0
            min_val_loss = np.min(val_loss) if len(val_loss) > 0 else 0
            overfitting_ratio = (
                final_train_loss / final_val_loss if final_val_loss > 0 else 0
            )
        else:
            final_val_loss = min_val_loss = overfitting_ratio = "N/A"

        # Create metrics text
        metrics_text = f"""📊 TRAINING METRICS

        🎯 Final Losses:
        • Training: {final_train_loss:.6f}
        • Validation: {final_val_loss if isinstance(final_val_loss, str) else f'{final_val_loss:.6f}'}

        📈 Loss Statistics:
        • Min Training Loss: {min_train_loss:.6f}
        • Max Training Loss: {max_train_loss:.6f}
        • Min Val Loss: {min_val_loss if isinstance(min_val_loss, str) else f'{min_val_loss:.6f}'}

        📉 Improvement:
        • Loss Reduction: {loss_improvement:.2f}%
        • Overfitting Ratio: {overfitting_ratio if isinstance(overfitting_ratio, str) else f'{overfitting_ratio:.3f}'}

        ⚙️ Training Info:
        • Total Epochs: {len(train_loss)}
        • Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}
        • Model: {self.cfg['model']['model_name'] if 'model_name' in self.cfg['model'] else 'Neural Inertial Tracking'}"""

        ax_metrics.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax_metrics.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # MSE subplot
        ax_mse = fig.add_subplot(gs[1, :])
        ax_mse.plot(
            epochs,
            train_mse,
            color="purple",
            linewidth=2,
            label="Training MSE",
            marker="o",
            markersize=3,
        )

        if val_loader is not None:
            ax_mse.plot(
                epochs,
                val_mse,
                color="orange",
                linewidth=2,
                label="Validation MSE",
                marker="s",
                markersize=3,
            )

        if test_loader is not None:
            ax_mse.plot(
                epochs,
                test_mse,
                color="brown",
                linewidth=2,
                label="Test MSE",
                marker="^",
                markersize=3,
            )

        ax_mse.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_mse.set_ylabel("MSE", fontsize=12, fontweight="bold")
        ax_mse.set_title("Mean Squared Error Progress", fontsize=14, fontweight="bold")
        ax_mse.grid(True, alpha=0.3, linestyle="--")
        ax_mse.legend(fontsize=10, framealpha=0.9)

        # Add final MSE metrics
        final_train_mse = train_mse[-1] if len(train_mse) > 0 else 0
        min_train_mse = np.min(train_mse) if len(train_mse) > 0 else 0
        mse_improvement = (
            ((train_mse[0] - final_train_mse) / train_mse[0] * 100)
            if len(train_mse) > 0 and train_mse[0] > 0
            else 0
        )

        mse_text = f"Final MSE: {final_train_mse:.6f} | Min MSE: {min_train_mse:.6f} | Improvement: {mse_improvement:.2f}%"
        ax_mse.text(
            0.02,
            0.98,
            mse_text,
            transform=ax_mse.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            osp.join(self.out_dir, "_epoch_loss.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save data files
        np.savetxt(
            osp.join(self.out_dir, "_epoch_loss.txt"),
            loss_all,
            delimiter=",",
            header=(
                "epoch,train_loss,val_loss,test_loss"
                if val_loader is not None and test_loader is not None
                else "epoch,train_loss"
            ),
        )
        np.savetxt(
            osp.join(self.out_dir, "_epoch_mse.txt"),
            mse_all,
            delimiter=",",
            header=(
                "epoch,train_mse,val_mse,test_mse"
                if val_loader is not None and test_loader is not None
                else "epoch,train_mse"
            ),
        )

        # Log completion with metrics
        success(f"✅ Training complete! Final training loss: {final_train_loss:.6f}")
        info(
            f"📊 Loss improvement: {loss_improvement:.2f}% | Final MSE: {final_train_mse:.6f}"
        )
        if val_loader is not None and not isinstance(overfitting_ratio, str):
            if overfitting_ratio > 1.1:
                warning(
                    f"⚠️ Potential overfitting detected (ratio: {overfitting_ratio:.3f})"
                )
            else:
                info(
                    f"✅ Good generalization (overfitting ratio: {overfitting_ratio:.3f})"
                )

    def _plot_online_adaptation_results(
        self, epoch_train_loss, epoch_train_mse, total_epochs
    ):
        """Enhanced plotting for online adaptation results."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Create figure with enhanced styling
        fig = plt.figure(num="Online Adaptation Analysis", dpi=150, figsize=(18, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

        # Main loss plot
        ax_main = fig.add_subplot(gs[0, 0])

        # Prepare data
        train_loss = np.array(epoch_train_loss)
        train_mse = np.array(epoch_train_mse)
        epochs = range(1, len(train_loss) + 1)

        # Enhanced plotting
        line_train = ax_main.plot(
            epochs,
            train_loss,
            color="#1f77b4",
            linewidth=2.5,
            label="Online Adaptation Loss",
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

        # Enhanced styling for main plot
        ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
        ax_main.set_title(
            "Online Adaptation Progress: Loss vs Epochs",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax_main.grid(True, alpha=0.3, linestyle="--")
        ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

        # Add trend analysis
        if len(train_loss) > 1:
            trend = np.polyfit(epochs, train_loss, 1)
            trend_line = np.poly1d(trend)(epochs)
            ax_main.plot(
                epochs,
                trend_line,
                "--",
                color="gray",
                alpha=0.7,
                linewidth=1.5,
                label="Trend",
            )

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_metrics.axis("off")

        # Calculate important metrics
        final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
        min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
        max_train_loss = np.max(train_loss) if len(train_loss) > 0 else 0
        loss_improvement = (
            ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
            if len(train_loss) > 0 and train_loss[0] > 0
            else 0
        )

        # Create metrics text
        metrics_text = f"""📊 ONLINE ADAPTATION METRICS
        🎯 Final Results:
        • Final Loss: {final_train_loss:.6f}
        • Min Loss: {min_train_loss:.6f}
        • Max Loss: {max_train_loss:.6f}

        📈 Performance:
        • Loss Reduction: {loss_improvement:.2f}%
        • Total Epochs: {total_epochs}
        • Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}

        ⚙️ Adaptation Info:
        • Method: Online Adaptation
        • Model: {self.cfg['model']['model_name'] if 'model_name' in self.cfg['model'] else 'Neural Inertial Tracking'}
        • Convergence: {'✅ Converged' if loss_improvement > 0 else '❌ No Improvement'}"""

        ax_metrics.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax_metrics.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        )

        # MSE subplot
        ax_mse = fig.add_subplot(gs[1, :])
        ax_mse.plot(
            epochs,
            train_mse,
            color="purple",
            linewidth=2,
            label="Online Adaptation MSE",
            marker="o",
            markersize=3,
        )

        ax_mse.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_mse.set_ylabel("MSE", fontsize=12, fontweight="bold")
        ax_mse.set_title("Mean Squared Error Progress", fontsize=14, fontweight="bold")
        ax_mse.grid(True, alpha=0.3, linestyle="--")
        ax_mse.legend(fontsize=10, framealpha=0.9)

        # Add final MSE metrics
        final_train_mse = train_mse[-1] if len(train_mse) > 0 else 0
        min_train_mse = np.min(train_mse) if len(train_mse) > 0 else 0
        mse_improvement = (
            ((train_mse[0] - final_train_mse) / train_mse[0] * 100)
            if len(train_mse) > 0 and train_mse[0] > 0
            else 0
        )

        mse_text = f"Final MSE: {final_train_mse:.6f} | Min MSE: {min_train_mse:.6f} | Improvement: {mse_improvement:.2f}%"
        ax_mse.text(
            0.02,
            0.98,
            mse_text,
            transform=ax_mse.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            osp.join(self.out_dir, "_online_adaptation_loss.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Save data files
        np.savetxt(
            osp.join(self.out_dir, "_online_adaptation_loss.txt"),
            train_loss[:, np.newaxis],
            delimiter=",",
            header="epoch,online_adaptation_loss",
        )
        np.savetxt(
            osp.join(self.out_dir, "_online_adaptation_mse.txt"),
            train_mse[:, np.newaxis],
            delimiter=",",
            header="epoch,online_adaptation_mse",
        )

        success(
            f"✅ Online adaptation plotting complete! Final loss: {final_train_loss:.6f}"
        )
        info(
            f"📊 Loss improvement: {loss_improvement:.2f}% | Final MSE: {final_train_mse:.6f}"
        )

    def _plot_offline_finetune_results(
        self, epoch_train_loss, epoch_val_loss, total_epochs
    ):
        """Enhanced plotting for offline finetuning results."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Create figure with enhanced styling
        fig = plt.figure(num="Offline Finetuning Analysis", dpi=150, figsize=(18, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], width_ratios=[3, 1])

        # Main loss plot
        ax_main = fig.add_subplot(gs[0, 0])

        # Prepare data
        train_loss = np.array(epoch_train_loss)
        epochs = range(1, len(train_loss) + 1)

        # Enhanced plotting
        line_train = ax_main.plot(
            epochs,
            train_loss,
            color="#1f77b4",
            linewidth=2.5,
            label="Training Loss",
            marker="o",
            markersize=4,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

        # Plot validation loss if available
        if epoch_val_loss and len(epoch_val_loss) > 0:
            val_loss = np.array(epoch_val_loss)
            line_val = ax_main.plot(
                epochs,
                val_loss,
                color="#d62728",
                linewidth=2.5,
                label="Validation Loss",
                marker="s",
                markersize=4,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

        # Enhanced styling for main plot
        ax_main.set_xlabel("Epoch", fontsize=14, fontweight="bold")
        ax_main.set_ylabel("Loss", fontsize=14, fontweight="bold")
        ax_main.set_title(
            "Offline Finetuning Progress: Loss vs Epochs",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax_main.grid(True, alpha=0.3, linestyle="--")
        ax_main.legend(fontsize=12, framealpha=0.9, loc="upper right")

        # Add trend analysis
        if len(train_loss) > 1:
            trend = np.polyfit(epochs, train_loss, 1)
            trend_line = np.poly1d(trend)(epochs)
            ax_main.plot(
                epochs,
                trend_line,
                "--",
                color="gray",
                alpha=0.7,
                linewidth=1.5,
                label="Trend",
            )

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 1])
        ax_metrics.axis("off")

        # Calculate important metrics
        final_train_loss = train_loss[-1] if len(train_loss) > 0 else 0
        min_train_loss = np.min(train_loss) if len(train_loss) > 0 else 0
        max_train_loss = np.max(train_loss) if len(train_loss) > 0 else 0
        loss_improvement = (
            ((train_loss[0] - final_train_loss) / train_loss[0] * 100)
            if len(train_loss) > 0 and train_loss[0] > 0
            else 0
        )

        if epoch_val_loss and len(epoch_val_loss) > 0:
            val_loss = np.array(epoch_val_loss)
            final_val_loss = val_loss[-1] if len(val_loss) > 0 else 0
            min_val_loss = np.min(val_loss) if len(val_loss) > 0 else 0
            overfitting_ratio = (
                final_train_loss / final_val_loss if final_val_loss > 0 else 0
            )
        else:
            final_val_loss = min_val_loss = overfitting_ratio = "N/A"

        # Create metrics text
        metrics_text = f"""📊 OFFLINE FINETUNING METRICS

🎯 Final Results:
• Training Loss: {final_train_loss:.6f}
• Validation Loss: {final_val_loss if isinstance(final_val_loss, str) else f'{final_val_loss:.6f}'}
• Min Training Loss: {min_train_loss:.6f}

📈 Performance:
• Loss Reduction: {loss_improvement:.2f}%
• Total Epochs: {total_epochs}
• Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}

⚙️ Finetuning Info:
• Method: Offline Finetuning
• Model: {self.cfg['model']['model_name'] if 'model_name' in self.cfg['model'] else 'Neural Inertial Tracking'}
• Overfitting Ratio: {overfitting_ratio if isinstance(overfitting_ratio, str) else f'{overfitting_ratio:.3f}'}"""

        ax_metrics.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax_metrics.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
        )

        # Learning rate subplot
        ax_lr = fig.add_subplot(gs[1, :])
        lr_values = [self.optimizer.param_groups[0]["lr"]] * len(
            epochs
        )  # Simplified - could track actual LR history
        ax_lr.plot(
            epochs,
            lr_values,
            color="orange",
            linewidth=2,
            label="Learning Rate",
            marker="o",
            markersize=3,
        )

        ax_lr.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax_lr.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
        ax_lr.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax_lr.grid(True, alpha=0.3, linestyle="--")
        ax_lr.legend(fontsize=10, framealpha=0.9)
        ax_lr.set_yscale("log")

        # Add final LR metrics
        final_lr = self.optimizer.param_groups[0]["lr"]
        lr_text = f"Final Learning Rate: {final_lr:.2e}"
        ax_lr.text(
            0.02,
            0.98,
            lr_text,
            transform=ax_lr.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

        # Adjust layout and save
        plt.tight_layout()
        fig.savefig(
            osp.join(self.out_dir, "_offline_finetune_loss.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Save data files
        np.savetxt(
            osp.join(self.out_dir, "_offline_finetune_loss.txt"),
            train_loss[:, np.newaxis],
            delimiter=",",
            header="epoch,offline_finetune_loss",
        )

        success(
            f"✅ Offline finetuning plotting complete! Final training loss: {final_train_loss:.6f}"
        )
        info(f"📊 Loss improvement: {loss_improvement:.2f}% | Final LR: {final_lr:.2e}")
        if (
            epoch_val_loss
            and len(epoch_val_loss) > 0
            and not isinstance(overfitting_ratio, str)
        ):
            if overfitting_ratio > 1.1:
                warning(
                    f"⚠️ Potential overfitting detected (ratio: {overfitting_ratio:.3f})"
                )
            else:
                info(
                    f"✅ Good generalization (overfitting ratio: {overfitting_ratio:.3f})"
                )

    def online_adaptation(
        self, train_loader, tester, resume_model, test_path_list, ate_thres
    ):
        # will always loop until the ave_ate meet requirements
        epoch_num = 0
        epoch_train_loss, epoch_train_mse, epoch_test_mse = [], [], []
        ave_ate = float("inf")

        # Fix 5: Initialize tracking variables for finetune stage
        if not hasattr(self, "best_train_loss") or self.best_train_loss == float("inf"):
            self.best_train_loss = float("inf")
            self.last_save_epoch = 0  # Add missing initialization
            info("Initialized tracking variables for finetune stage")

        # freeze subt base model params
        self.model = freeze_backbone_parameters(self.model)

        # add lora params to base model (if available)
        if LORA_AVAILABLE:
            if hasattr(self.model, "module"):
                lora.replace_layers(self.model.module)
            else:
                lora.replace_layers(self.model)
            # add lora params to optimizer
            new_parameters = []
            for param in self.model.parameters():
                if param.requires_grad:
                    new_parameters.append(param)
            for param_group in self.optimizer.param_groups:
                param_group["params"].extend(new_parameters)
        else:
            logging.warning("LoRA not available - using standard finetuning approach")
        # self.model = trainable_BN_parameters(self.model)
        # while(ave_ate>ate_thres):
        all_length = 23997
        current_frame = 3000  # 20s
        time_buffer = []
        while True:
            # logging.warn(f"train_data_path_list: {test_path_list}")
            epoch_num = epoch_num + 1
            rich_logger.training_header(
                epoch_num, 100, "Online Adaptation"
            )  # Use 100 as max epochs for display
            torch.cuda.synchronize()
            start_t = time.time()
            # if epoch_num==1:
            train_attr_dict = self.train_step(
                train_loader, epoch_num, fix_backbone=True, current_frame=current_frame
            )
            # if((ave_ate-ate_thres)>1):
            #     train_attr_dict = self.train_step(train_loader, epoch_num, fix_backbone=False)
            # else:
            #     train_attr_dict = self.train_step(train_loader, epoch_num, fix_backbone=True)
            torch.cuda.synchronize()
            end_t = time.time()
            epoch_time = end_t - start_t
            time_buffer.append(epoch_time)
            ratio = current_frame / all_length
            if sum(time_buffer) > int((current_frame + 200 - 4000) / 200):
                current_frame += 200
            train_loss = np.average(train_attr_dict["losses"])
            train_loss_mse = np.mean(
                (train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2
            )
            instance_per_second = len(train_loader.dataset) / epoch_time

            if self.log:
                write_wandb("online_adapt/time", epoch_time, epoch_num, self.local_rank)
                write_wandb(
                    "online_adapt/instance_time",
                    instance_per_second,
                    epoch_num,
                    self.local_rank,
                )
                write_wandb(
                    "online_adapt/epoch_time", epoch_time, epoch_num, self.local_rank
                )
                write_wandb(
                    "online_adapt/train_loss", train_loss, epoch_num, self.local_rank
                )
                write_wandb(
                    "online_adapt/train_loss_mse",
                    train_loss_mse,
                    epoch_num,
                    self.local_rank,
                )

            epoch_train_loss.append(train_loss)  # The loss for each epoch is the mean of all batch losses
            epoch_train_mse.append(train_loss_mse)
            info(f'Average loss: {np.average(train_attr_dict["losses"]):.6f}')
            self.scheduler.step(np.average(train_attr_dict["losses"]))
            # self.scheduler.step()
            # print('step count:', self.scheduler._step_count)
            # if self.scheduler._step_count > 100:
            #     exit()
            # print('fine-tune lr:', self.scheduler.get_last_lr())
            logging.info(
                f"current learning rate: {self.optimizer.param_groups[0]['lr']}"
            )
            if self.optimizer.param_groups[0]["lr"] < 1.1e-6:
                break
            # if self.local_rank == 0:
            #     wandb.log(wandb_log_dict)

            # Optimized checkpoint saving for finetune - save less frequently for speed
            self.save_checkpoint_strategy(epoch_num, train_loss, None, None)

            resume_model = self.model
            resume_model.eval()

            # Safety check for tester
            if tester is None or not hasattr(tester, "test"):
                warning("Tester is not available or invalid. Skipping test evaluation.")
                ave_ate = float("inf")  # Default value
            else:
                try:
                    all_metrics = tester.test(
                        test_path_list, epoch_num, resume_model, ratio=ratio
                    )
                    ave_ate = all_metrics["all_traj"]["avg_ate"]
                except Exception as e:
                    warning(f"Error during test evaluation: {e}")
                    ave_ate = float("inf")  # Default value

            # Log training statistics (replaced log_setting with standard logging)
            logging.info("Training Statistics:")
            training_statics = {
                "Epoch_num": epoch_num,
                "epoch_time": epoch_time,
                "Learning_rate": self.optimizer.param_groups[0]["lr"],
                "Train_loss": np.average(train_attr_dict["losses"]),
                #  "Batch_normal_weight": resume_model.module.model.resnet_post_pro[1].weight[0],
                #  "Batch_normal_bias": resume_model.module.model.resnet_post_pro[1].bias[0],
                "avg_atr": ave_ate,
            }
            logging.info(f"Training Statistics: {training_statics}")

        # Enhanced plotting for online adaptation
        if len(epoch_train_loss) > 0:
            self._plot_online_adaptation_results(
                epoch_train_loss, epoch_train_mse, epoch_num
            )

        return resume_model

    def offline_finetune(
        self, train_loader, val_loader=None, test_loader=None, max_epochs=20
    ):
        """
        Simple offline finetuning - much more stable than online adaptation

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            max_epochs: Maximum number of finetuning epochs
        """
        logging.info("Starting offline finetuning...")

        # Freeze backbone parameters for finetuning (only once)
        if not hasattr(self, "_backbone_frozen"):
            self.model = freeze_backbone_parameters(self.model)
            logging.info("Backbone parameters frozen for finetuning")
            self._backbone_frozen = True

        # Track finetuning progress
        epoch_train_loss, epoch_val_loss = [], []
        best_train_loss = float("inf")
        best_val_loss = float("inf") if val_loader else float("inf")
        last_save_epoch = 0  # Track last save epoch for emergency saves

        for epoch in range(1, max_epochs + 1):
            logging.info(
                f"-------------- Offline Finetune Epoch {epoch}/{max_epochs} ---------------"
            )

            # Training step
            train_attr_dict = self.train_step(train_loader, epoch, fix_backbone=True)
            train_loss = np.average(train_attr_dict["losses"])
            train_mse = np.mean(
                (train_attr_dict["targets"] - train_attr_dict["preds"]) ** 2
            )
            epoch_train_loss.append(train_loss)

            # Validation step (if available)
            val_loss = None
            if val_loader:
                val_attr_dict = self.inference_step(val_loader, epoch)
                val_loss = np.average(val_attr_dict["losses"])
                val_mse = np.mean(
                    (val_attr_dict["targets"] - val_attr_dict["preds"]) ** 2
                )
                epoch_val_loss.append(val_loss)

                if self.log:
                    write_wandb("finetune/val_loss", val_loss, epoch, self.local_rank)
                    write_wandb("finetune/val_mse", val_mse, epoch, self.local_rank)

            # Learning rate scheduling
            if val_loader and val_loss is not None:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step(train_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            if self.log:
                write_wandb("finetune/train_loss", train_loss, epoch, self.local_rank)
                write_wandb("finetune/train_mse", train_mse, epoch, self.local_rank)
                write_wandb(
                    "finetune/learning_rate", current_lr, epoch, self.local_rank
                )

            logging.info(
                f"Epoch {epoch}: train_loss={train_loss:.6f}, lr={current_lr:.6f}"
            )
            if val_loss:
                logging.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")

            # Use unified checkpoint saving strategy
            self.save_checkpoint_strategy(epoch, train_loss, val_loss, None)

            # Early stopping conditions
            if current_lr < 1e-6:
                logging.info(
                    f"Learning rate too small ({current_lr:.6f}), stopping finetuning"
                )
                break

            if val_loader and val_loss and epoch > 10:
                # Check if validation loss hasn't improved for 5 epochs
                if len(epoch_val_loss) >= 5:
                    recent_val_losses = epoch_val_loss[-5:]
                    if all(
                        recent_val_losses[i] >= recent_val_losses[i - 1]
                        for i in range(1, 5)
                    ):
                        logging.info(
                            "Validation loss not improving for 5 epochs, stopping finetuning"
                        )
                        break

        # Always save final model (only from rank 0)
        if self.local_rank == 0:
            save_model(
                self.out_dir,
                max_epochs,
                self.model,
                self.optimizer,
                self.use_multi_gpu,
                self.local_rank,
            )
            logging.info(f"Final finetune model saved at epoch {max_epochs}")
        else:
            logging.info(f"Final model save skipped (rank {self.local_rank})")

        logging.info("Offline finetuning completed!")

        # Enhanced plotting for offline finetuning
        if len(epoch_train_loss) > 0:
            self._plot_offline_finetune_results(
                epoch_train_loss, epoch_val_loss, max_epochs
            )

        return self.model

    def save_checkpoint_strategy(self, epoch, train_loss, val_loss, test_loss):
        """
        Unified checkpoint saving strategy with clear logic and proper rank handling.

        Args:
            epoch: Current epoch number
            train_loss: Current training loss
            val_loss: Current validation loss (None if no validation)
            test_loss: Current test loss (None if no test)
        """
        # Only save from rank 0 to avoid multiple checkpoints
        if self.local_rank != 0:
            return

        should_save = False
        save_reason = ""

        # Always save the first epoch
        if self.best_train_loss == float("inf"):
            self.best_train_loss = train_loss
            should_save = True
            save_reason = "first_epoch"
            success(f"First epoch checkpoint saved - loss: {train_loss:.6f}")

        # Save on validation loss improvement
        elif val_loss is not None and val_loss < self.best_val_loss:
            improvement = (self.best_val_loss - val_loss) / self.best_val_loss
            if improvement > self.significant_improvement_threshold:
                self.best_val_loss = val_loss
                should_save = True
                save_reason = f"improved_val_loss"
                success(
                    f"Validation loss improved by {improvement:.3f} - saving checkpoint"
                )

        # Save on training loss improvement (with lower threshold)
        elif train_loss < self.best_train_loss:
            improvement = (self.best_train_loss - train_loss) / self.best_train_loss
            if improvement > 0.005:  # Lower threshold to 0.5% for more frequent saves
                self.best_train_loss = train_loss
                should_save = True
                save_reason = f"improved_train_loss"
                success(
                    f"Training loss improved by {improvement:.3f} - saving checkpoint"
                )

        # Save if validation is better than best training loss
        elif val_loss is not None and val_loss < self.best_train_loss:
            self.best_train_loss = val_loss
            should_save = True
            save_reason = "validation_better_than_train"
            success(
                f"Validation loss {val_loss:.6f} better than best train loss {self.best_train_loss:.6f} - saving checkpoint"
            )

        # Regular saves every 10 epochs (moved out of else block)
        if epoch % 10 == 0 and not should_save:
            should_save = True
            save_reason = f"regular_save_every_10_epochs"
            info(f"Regular checkpoint save every 10 epochs - epoch {epoch}")

        # Emergency save if no checkpoint for 5 epochs
        elif epoch - self.last_save_epoch >= 5 and not should_save:
            should_save = True
            save_reason = f"emergency_save_after_5_epochs"
            warning(
                f"Emergency checkpoint save after 5 epochs without saving - epoch {epoch}"
            )

        # Save the checkpoint
        if should_save:
            save_model(
                self.out_dir,
                epoch,
                self.model,
                self.optimizer,
                self.use_multi_gpu,
                self.local_rank,
                save_reason,
            )
            self.last_save_epoch = epoch
            success(f"Checkpoint saved: {save_reason} at epoch {epoch}")
        else:
            debug(
                f"Checkpoint NOT saved at epoch {epoch}. should_save={should_save}, best_train_loss={self.best_train_loss:.6f}, current_train_loss={train_loss:.6f}, epoch_diff={epoch - self.last_save_epoch}"
            )
