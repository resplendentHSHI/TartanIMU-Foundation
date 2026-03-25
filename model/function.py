import logging
import pdb
from collections import Counter

import torch
from model.losses import (
    efficient_multi_head_smooth_loss,
    get_sequence_smooth_loss,
    multi_head_smooth_loss,
)


def fun_train_forward(cfg, model, batch, start_cov_epochs, epoch):
    """
    Training forward pass with proper motion type handling.

    Args:
        cfg (dict): Configuration dictionary
        model (nn.Module): Neural network model
        batch (tuple): (features, targets, aux_data, motion_type)
        start_cov_epochs (int): Epoch to start covariance prediction
        epoch (int): Current epoch

    Returns:
        tuple: (predictions, covariances, targets, loss)
    """
    model_name = cfg["model"]["model_name"]
    feat, targ, _, motion_type = (
        batch  # feat: [batch_size, seq_len, 6, frames], targ: [batch_size, seq_len, 3]
    )

    if epoch <= start_cov_epochs:  # Before covariance prediction
        pred_multi_head = model(feat, motion_type)  # Use motion_type for optimization
        pred_multi_cov = {}
        for key, value in pred_multi_head.items():
            pred_multi_cov[key] = torch.zeros_like(value)
    else:  # Predict covariance
        logging.info("Starting covariance prediction training")
        pred_multi_head, pred_multi_cov = model(feat, motion_type, predict_cov=True)

    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
    multi_head_mask = {}

    for motion_id, motion_name in motion_types.items():
        multi_head_mask[motion_name] = motion_type == motion_id

    for key in pred_multi_head:
        output_dim = cfg["model_param"]["output_dim"]  # 3
        pred_multi_head[key] = pred_multi_head[key][
            :,
            cfg["train"]["predict_start"] :,
        ]  # ['predict_start']->0   pred: [1, 10, 3]
        pred_multi_cov[key] = pred_multi_cov[key][
            :,
            cfg["train"]["predict_start"] :,
        ]  # pred_cov: [1, 10, 3]
        targ = targ[:, cfg["train"]["predict_start"] :, 0:output_dim]  # [1, 10, 3]

    # Note: Velocity scaling is now handled by learnable parameters in OutputHead
    # if cfg['model']['pred_velocity']:
    #     for key in pred_multi_head:
    #         pred_multi_head[key] = pred_multi_head[key] * cfg['model_param']['window_time']
    #         pred_multi_cov[key] = torch.log(cfg['model_param']['window_time'] * torch.ones_like(pred_multi_cov[key])) + pred_multi_cov[key]

    # print("cfg['data']['use_local_coord']:",cfg['data']['use_local_coord'])

    loss = multi_head_smooth_loss(
        pred_multi_head,
        pred_multi_cov,
        targ,
        epoch,
        multi_head_mask,
        start_cov_epochs,
        cfg["data"]["use_local_coord"],
    )

    # loss = get_sequence_smooth_loss(pred_multi_head, pred_multi_cov, targ, epoch, start_cov_epochs)

    for key in pred_multi_head:
        all_b = pred_multi_head[key].size(0) * pred_multi_head[key].size(1)  # 1*10
        pred_multi_head[key] = pred_multi_head[key].contiguous().view(all_b, -1)  # 10*3
        pred_multi_cov[key] = pred_multi_cov[key].contiguous().view(all_b, -1)  # 10*3
        targ = targ.contiguous().view(all_b, -1)  # 10*3
    return pred_multi_head[key], pred_multi_cov[key], targ, loss


def fun_train_forward_efficient(cfg, model, batch, start_cov_epochs, epoch):
    """
    Efficient training forward pass that only computes needed heads.
    """
    model_name = cfg["model"]["model_name"]
    feat, targ, _, motion_type = batch

    needed_heads = get_active_heads(cfg, motion_type)

    # # Determine which heads are needed
    # needed_heads = get_needed_heads_from_motion_type(motion_type)

    if epoch <= start_cov_epochs:
        # Only compute predictions for needed heads
        pred_multi_head = model(feat, motion_type, compute_all_heads=False)
        pred_multi_cov = {}
        for key, value in pred_multi_head.items():
            pred_multi_cov[key] = torch.zeros_like(value)
    else:
        pred_multi_head, pred_multi_cov = model(
            feat, motion_type, predict_cov=True, compute_all_heads=False
        )

    # Create masks only for needed heads
    multi_head_mask = {}
    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}

    for motion_id, motion_name in motion_types.items():
        if motion_name in needed_heads and motion_name in pred_multi_head:
            multi_head_mask[motion_name] = motion_type == motion_id

    # Process predictions
    for key in pred_multi_head:
        output_dim = cfg["model_param"]["output_dim"]
        pred_multi_head[key] = pred_multi_head[key][
            :,
            cfg["train"]["predict_start"] :,
        ]
        pred_multi_cov[key] = pred_multi_cov[key][
            :,
            cfg["train"]["predict_start"] :,
        ]
        targ = targ[:, cfg["train"]["predict_start"] :, 0:output_dim]

    # Note: Velocity scaling is now handled by learnable parameters in OutputHead
    # if cfg['model']['pred_velocity']:
    #     for key in pred_multi_head:
    #         pred_multi_head[key] = pred_multi_head[key] * cfg['model_param']['window_time']
    #         pred_multi_cov[key] = torch.log(cfg['model_param']['window_time'] * torch.ones_like(pred_multi_cov[key])) + pred_multi_cov[key]

    loss = efficient_multi_head_smooth_loss(
        pred_multi_head,
        pred_multi_cov,
        targ,
        epoch,
        multi_head_mask,
        start_cov_epochs,
        cfg["data"]["use_local_coord"],
    )

    for key in pred_multi_head:
        all_b = pred_multi_head[key].size(0) * pred_multi_head[key].size(1)  # 1*10
        pred_multi_head[key] = pred_multi_head[key].contiguous().view(all_b, -1)  # 10*3
        pred_multi_cov[key] = pred_multi_cov[key].contiguous().view(all_b, -1)  # 10*3
        targ = targ.contiguous().view(all_b, -1)  # 10*3

    return pred_multi_head[key], pred_multi_cov[key], targ, loss


def get_needed_heads_from_motion_type(motion_type):
    """Extract needed heads from motion type tensor."""
    if isinstance(motion_type, torch.Tensor):
        unique_types = torch.unique(motion_type).tolist()
    else:
        unique_types = [motion_type]

    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}
    needed_heads = []

    for motion_id in unique_types:
        if motion_id in motion_types:
            needed_heads.append(motion_types[motion_id])

    return needed_heads


def get_active_heads(cfg, motion_type):
    """Return heads to train; prefer config over batch inference."""
    ah = cfg.get("active_heads", None)
    if ah is not None and len(ah) > 0:
        return list(ah)
    # fallback (your existing behavior)
    return get_needed_heads_from_motion_type(motion_type)


def fun_test_forward(cfg, model, batch, start_cov_epochs, epoch):
    """
    Testing forward pass with proper motion type handling.

    Args:
        cfg (dict): Configuration dictionary
        model (nn.Module): Neural network model
        batch (tuple): (features, targets, orientations, motion_type)
        start_cov_epochs (int): Epoch to start covariance prediction
        epoch (int): Current epoch

    Returns:
        tuple: (predictions, covariances, targets, orientations, loss)
    """
    model_name = cfg["model"]["model_name"]
    feat, targ, orien, motion_type = batch

    # Determine the most common motion type in the batch
    frequency = Counter(motion_type.tolist())
    most_motion_pattern, count = frequency.most_common(1)[0]
    motion_types = {1: "car", 2: "dog", 3: "drone", 4: "human"}

    if most_motion_pattern not in motion_types:
        raise ValueError(f"Unknown motion type: {most_motion_pattern}")

    motion_dynamic = motion_types[most_motion_pattern]

    if epoch <= start_cov_epochs:
        pred_multi_head = model(feat, motion_type)  # Use motion_type for optimization
        pred = pred_multi_head[motion_dynamic]
        pred_cov = torch.zeros_like(pred)
    else:
        pred_multi_head, pred_multi_cov = model(
            feat, motion_type, predict_cov=True
        )  # Use motion_type for optimization
        pred = pred_multi_head[motion_dynamic]
        pred_cov = pred_multi_cov[motion_dynamic]
    output_dim = cfg["model_param"]["output_dim"]
    pred = pred[
        :,
        -1,
    ]  # Test output: last prediction result of the 10-sequence segment (B,3)
    pred_cov = pred_cov[
        :,
        -1,
    ]  # Variance of the last prediction of the 10-sequence segment (B,3)
    targ = targ[:, -1, 0:output_dim]  # (B,3)

    # Note: Velocity scaling is now handled by learnable parameters in OutputHead
    # if cfg['model']['pred_velocity']:
    #     pred = pred * cfg['model_param']['window_time']
    #     pred_cov = torch.log(cfg['model_param']['window_time'] * torch.ones_like(pred_cov)) + pred_cov
    imu_freq = int(cfg["data"]["imu_freq"])
    orien = orien[
        :,
        -imu_freq:,
    ]  # Take the ground truth rotation of the last window
    if cfg["model"]["pred_velocity"]:
        pred = pred * cfg["model_param"]["window_time"]
        pred_cov = (
            torch.log(cfg["model_param"]["window_time"] * torch.ones_like(pred_cov))
            + pred_cov
        )

        # Apply velocity smoothing to reduce jittering
        from model.losses import smooth_velocity_predictions

        pred = smooth_velocity_predictions(pred, window_size=3)

    # loss=multi_head_smooth_loss(pred_multi_head, pred_multi_cov, targ, epoch, multi_head_mask, start_cov_epochs, cfg['data']['use_local_coord'])
    loss = get_sequence_smooth_loss(pred, pred_cov, targ, epoch, start_cov_epochs)

    return pred, pred_cov, targ, orien, loss
