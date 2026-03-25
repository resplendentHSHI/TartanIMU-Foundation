import logging
import pdb

import torch
import torch.nn.functional as F

EPSILON = 1e-7


def loss_distribution_diag(pred, pred_cov, targ):
    # Bounded covariance-based diagonal Gaussian NLL (up to constant)
    pred_cov_clamped = torch.clamp(pred_cov, min=-10.0, max=10.0)
    exp_term = torch.exp(2 * pred_cov_clamped)
    exp_term = torch.clamp(exp_term, min=1e-6, max=1e6)

    squared_error = (pred - targ).pow(2)
    loss = squared_error / (2 * exp_term) + pred_cov_clamped
    loss = torch.clamp(loss, min=1e-8)
    return loss


def loss_cum_distribution_diag(pred, pred_cov, targ):
    # Kept for compatibility (unused after simplification)
    pred_cov_clamped = torch.clamp(pred_cov, min=1e-6, max=1e6)
    loss = ((pred - targ).pow(2)) / (2 * pred_cov_clamped) + 0.5 * torch.log(
        pred_cov_clamped
    )
    loss = torch.clamp(loss, min=1e-8)
    return loss


# to do rewrite loss function.
def get_sequence_smooth_loss(pred, pred_cov, targ, epoch, start_cov_epoch):
    """
    Simplified sequence loss:
    - Before start_cov_epoch: per-step MSE
    - After start_cov_epoch: bounded covariance loss (diagonal Gaussian NLL style)
    """
    if epoch <= start_cov_epoch:
        loss = (pred - targ).pow(2)
    else:
        loss = loss_distribution_diag(pred, pred_cov, targ)
    return torch.mean(loss)


def multi_head_smooth_loss(
    multi_head_pred,
    multi_head_cov,
    targ,
    epoch,
    multi_head_mask,
    start_cov_epoch,
    use_local_coord,
):
    multi_head_loss = {}
    total_loss = 0
    for key in multi_head_pred:
        multi_head_loss[key] = single_head_mask_loss(
            multi_head_pred[key],
            multi_head_cov[key],
            targ,
            epoch,
            multi_head_mask[key],
            start_cov_epoch,
            use_local_coord,
        )
        total_loss = multi_head_loss[key] + total_loss
    return total_loss


def efficient_multi_head_smooth_loss(
    multi_head_pred,
    multi_head_cov,
    targ,
    epoch,
    multi_head_mask,
    start_cov_epochs,
    use_local_coord,
):
    """
    Efficient multi-head smooth loss with debug logging.
    """
    # Debug logging every 10 epochs
    if epoch % 10 == 0:
        print(f"\n🔍 DEBUG Epoch {epoch}: Loss Analysis")
        print(f"  multi_head_pred keys: {list(multi_head_pred.keys())}")
        print(f"  multi_head_mask keys: {list(multi_head_mask.keys())}")

        for key, pred in multi_head_pred.items():
            if key in multi_head_mask:
                mask = multi_head_mask[key]
                active_samples = mask.sum().item()
                if active_samples > 0:
                    masked_pred = pred[mask]
                    masked_targ = targ[mask]
                    print(f"  {key} head - active samples: {active_samples}")
                    print(
                        f"  {key} head - pred range: [{masked_pred.min():.6f}, {masked_pred.max():.6f}]"
                    )
                    print(
                        f"  {key} head - target range: [{masked_targ.min():.6f}, {masked_targ.max():.6f}]"
                    )

    total_loss = 0
    num_active_heads = 0

    for key, pred in multi_head_pred.items():
        if key in multi_head_mask:
            mask = multi_head_mask[key]
            if mask.sum() > 0:
                head_loss = single_head_mask_loss(
                    pred,
                    multi_head_cov.get(key, torch.zeros_like(pred)),
                    targ,
                    epoch,
                    mask,
                    start_cov_epochs,
                    use_local_coord,
                )

                # Debug logging
                if epoch % 10 == 0:
                    print(f"  {key} head loss: {head_loss.item():.6f}")

                total_loss += head_loss
                num_active_heads += 1

    if num_active_heads > 0:
        total_loss = total_loss / num_active_heads

    # Debug logging for total loss
    if epoch % 10 == 0:
        print(f"  Total averaged loss: {total_loss.item():.6f}")

    return total_loss


def smooth_transition_weight(epoch, start_cov_epoch, transition_epochs=10):
    """
    Smooth transition weight for covariance training.
    Returns a value between 0 and 1 that gradually increases over transition_epochs.
    """
    if epoch <= start_cov_epoch:
        return 0.0
    elif epoch >= start_cov_epoch + transition_epochs:
        return 1.0
    else:
        # Linear interpolation
        progress = (epoch - start_cov_epoch) / transition_epochs
        return progress


def single_head_mask_loss(
    pred, pred_cov, targ, epoch, mask, start_cov_epoch, use_local_coord=False
):
    """
    Simplified masked loss per head:
    - MSE before covariance training
    - Covariance-based loss after, with smooth transition
    Absolute/cumulative terms are removed for simplicity.
    """
    # Smooth transition for covariance
    cov_weight = smooth_transition_weight(epoch, start_cov_epoch, transition_epochs=5)

    # Debug logging for transition monitoring
    if epoch % 5 == 0 and epoch >= start_cov_epoch - 5:
        print(
            f"🔧 Epoch {epoch}: cov_weight={cov_weight:.3f}, start_cov_epoch={start_cov_epoch}"
        )

    if use_local_coord:
        # Keep velocity loss path for local coord (not used in current cfg)
        loss_covariance = single_head_velocity_loss(pred, pred_cov, targ)
        loss = loss_covariance["loss"]
        # Apply mask
        for i, m in enumerate(mask):
            if m == 0:
                loss[i, :, :] = 0
        return torch.mean(loss)

    # Per-step MSE
    mse_loss = (pred - targ).pow(2)

    # Add scale-aware loss for velocity prediction (when use_local_coord=True)
    if use_local_coord:
        # Compute scale error (ratio of predicted to target magnitudes)
        pred_mag = torch.norm(pred, dim=-1, keepdim=True)
        targ_mag = torch.norm(targ, dim=-1, keepdim=True)

        # Avoid division by zero
        targ_mag = torch.clamp(targ_mag, min=1e-6)
        scale_ratio = pred_mag / targ_mag

        # Penalize scale errors (log-scale loss)
        scale_loss = torch.log(scale_ratio + 1e-6).pow(2)

        # Combine MSE and scale loss
        mse_loss = mse_loss + 0.1 * scale_loss

    # Apply mask to MSE
    for i, m in enumerate(mask):
        if m == 0:
            mse_loss[i, :, :] = 0

    if cov_weight > 0:
        cov_loss = loss_distribution_diag(pred, pred_cov, targ)
        # Apply mask to covariance loss
        for i, m in enumerate(mask):
            if m == 0:
                cov_loss[i, :, :] = 0
        # Weighted combination
        loss = (1 - cov_weight) * mse_loss + cov_weight * cov_loss
        # Debug logging for loss components
        if epoch % 5 == 0 and epoch >= start_cov_epoch - 5:
            mse_mean = torch.mean(mse_loss).item()
            cov_mean = torch.mean(cov_loss).item()
            final_mean = torch.mean(loss).item()
            print(
                f"  📊 Loss components - MSE: {mse_mean:.6f}, Cov: {cov_mean:.6f}, Final: {final_mean:.6f}"
            )
    else:
        loss = mse_loss

    return torch.mean(loss)


def L2(dist):
    error = dist.pow(2)
    return torch.mean(error)


def L1(dist):
    error = (dist).abs().mean()
    return error


def Huber(dist, delta=0.005):
    error = torch.nn.functional.huber_loss(
        dist, torch.zeros_like(dist, device=dist.device), delta=delta, reduction="none"
    )
    return error


def motion_loss_(fc, pred, targ):
    # loss1 = F.l1_loss(pred, targ)
    dist = pred - targ
    loss = fc(dist)
    # if torch.isnan(loss):
    #     pdb.set_trace()
    #     print(loss)
    return loss, dist


def diag_ln_cov_loss(dist, pred_cov, use_epsilon=False):
    error = (dist).pow(2)
    if use_epsilon:
        l = (error / pred_cov) + torch.log(pred_cov + EPSILON)
    else:
        l = (error / pred_cov) + torch.log(pred_cov)
    return l.mean()


def single_head_velocity_loss(pred, pred_cov, targ):
    # TODO: Need to put the following parameters in the yaml file
    confs = {
        "loss": "L1",
        "propcov": False,
        "cov_weight": 1e-4,
        "covaug": False,
        "weight": 20,  # 20
    }
    # Decouple velocity prediction for xyz three axes
    ## The state loss for evaluation
    loss, cov_loss = torch.zeros_like(pred, device=pred.device), torch.zeros_like(
        pred, device=pred.device
    )
    loss_fc = loss_fc_list[confs["loss"]]
    vel_loss, vel_dist = motion_loss_(loss_fc, pred, targ)

    # Apply the covariance loss
    if confs["propcov"]:
        # velocity covariance.
        cov = pred_cov
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs["cov_weight"] * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs["cov_weight"] * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs["weight"] * vel_loss
    return {"loss": loss, "cov_loss": cov_loss}


loss_fc_list = {
    "L2": L2,
    "L1": L1,
    "diag_cov_ln": diag_ln_cov_loss,
    "Huber_loss005": lambda dist: Huber(dist, delta=0.005),
    "Huber_loss05": lambda dist: Huber(dist, delta=0.05),
}


def smooth_velocity_predictions(velocities, window_size=3):
    """
    Simple post-processing function to smooth velocity predictions and reduce jittering.

    Args:
        velocities: [B, T, 3] or [T, 3] velocity predictions
        window_size: Size of smoothing window (odd number recommended)

    Returns:
        Smoothed velocities with same shape as input
    """
    import torch.nn.functional as F

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Handle different input shapes
    if velocities.dim() == 2:
        velocities = velocities.unsqueeze(0)  # [T, 3] -> [1, T, 3]
        single_batch = True
    else:
        single_batch = False

    # Simple moving average smoothing
    pad_size = window_size // 2
    padded = F.pad(velocities, (0, 0, pad_size, pad_size), mode="replicate")
    smoothed = F.avg_pool1d(
        padded.transpose(1, 2),  # [B, C, T+2*pad]
        kernel_size=window_size,
        stride=1,
        padding=0,
    ).transpose(
        1, 2
    )  # [B, T, C]

    if single_batch:
        smoothed = smoothed.squeeze(0)  # [1, T, 3] -> [T, 3]

    return smoothed
