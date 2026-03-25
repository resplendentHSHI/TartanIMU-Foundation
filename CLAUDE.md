# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TartanIMU is a neural inertial tracking system that predicts velocity (and optionally covariance) from IMU data (6-channel: accelerometer + gyroscope). It uses a **multi-head Foundation Model** architecture supporting multiple motion types (car, dog, drone, human) simultaneously.

## Commands

### Training
```bash
# Single GPU training
python main_net.py --config config/datasets/tartanimu/humanoid_train.yaml

# Multi-GPU distributed training (uses torchrun)
torchrun --nproc_per_node=<NUM_GPUS> main_net.py --config config/datasets/tartanimu/humanoid_train.yaml

# Resume/finetune from checkpoint
python main_net.py --config config/datasets/tartanimu/humanoid_train.yaml --resume_from /path/to/checkpoint.pt
```

### Testing
```bash
python main_net.py --config config/datasets/tartanimu/humanoid_test.yaml
```

### Minimal Example (inference + fine-tuning)
```bash
python minimal_example.py --config <config.yaml> --model <checkpoint.pth> --test_data <data.npz> --motion_type human
```

### Install Dependencies
```bash
pip install -r requirement.txt
```

## Architecture

### Model Pipeline

`main_net.py` is the entry point. It loads config, builds data loaders, and delegates to `train.py` (Trainer) or `test.py` (tester) based on `schemes.train`/`schemes.test` in the YAML config.

### Foundation Model (`model/model_lstm.py`)

The core model is a **shared ResNet-LSTM trunk** with **per-motion-type output heads**:

```
IMU input [B, seq_len, 6, frames]
    → ResNet1D backbone (residual conv blocks)
    → LSTM temporal encoder
    → Per-motion-type OutputHead (velocity head + optional covariance head)
```

- `FoundationModel` wraps a `ResNetLSTMSeqNet` trunk + `nn.ModuleDict` of `OutputHead`s
- Each `OutputHead` has `output_block1` (velocity) and `output_block2` (covariance)
- During training, only heads for motion types present in the batch are computed (`compute_all_heads=False`)
- Covariance prediction activates after `start_cov_epochs` (configured in YAML)

### Motion Type System

Motion types are integer-coded: `{1: "car", 2: "dog", 3: "drone", 4: "human"}` (defined in `utils/constants.py` as `MOTION_TYPES`). The batch carries a motion type tensor that determines which heads to compute and which loss masks to apply.

### Forward Pass Variants (`model/function.py`)

- `fun_train_forward` / `fun_train_forward_efficient`: Training forward pass with multi-head masked loss
- `fun_test_forward`: Test forward pass, takes only the last prediction of each sequence segment

### Configuration System

Two-level YAML config: a dataset config (e.g., `config/datasets/tartanimu/humanoid_train.yaml`) references a model config via `model.model_yaml` (e.g., `config/resnet_lstm_multihead.yaml`). The `configer.py:build_model()` merges them via `update_recursive`. Key config sections:

- `schemes`: Controls train/test/online_adaption mode
- `data`: Dataset paths per motion type, IMU freq, gravity alignment settings
- `model`: Model name, model_yaml path, pred_velocity flag
- `train`: Batch size, epochs, start_cov_epochs, optimizer, scheduler, active_heads
- `augment`: IMU noise augmentation parameters

### Data Format

Data is stored as `.npz` files organized by motion type: `<motion_type>/{train,val,test}/<sequence_id>/<file>.npz`. The dataset class `dataloader/dataset_AirLab.py` (`ResNetLSTMSeqToSeqDataset`) handles loading and windowing into sequences of length `train.seq_len`.

### Loss Functions (`model/losses.py`)

Multi-head masked loss: each head's loss is computed only on samples matching its motion type. After `start_cov_epochs`, a covariance-aware loss is used for uncertainty estimation.

### Key Config Flags

- `train.active_heads`: List of motion types to train (e.g., `["human"]`). If set, overrides batch-inferred heads.
- `model.pred_velocity: True`: Model predicts velocity (scaled by `window_time`), not displacement.
- `data.use_local_coord`: Whether to use local coordinates for loss computation.
- `train.use_multi_gpu`: Enables DDP with NCCL backend and SyncBatchNorm.
- `train.use_amp`: Enables automatic mixed precision.

### Logging

Uses `utils/rich_logging.py` for colored console output (Rich library) and WandB for experiment tracking.
