# autoresearch — TartanIMU Optimization

This is an experiment to have the LLM autonomously optimize a neural inertial tracking model's validation loss and training speed.

## Objective

Improve **validation loss** and **training runtime** on the new_lamar/ios dataset using the 85/15 train/val split. The baseline is:

- **Best val loss: 0.1090** (MSE, epoch 12 of 20)
- **Epoch time: ~120 seconds** (~40 min for 20 epochs)
- **Config: `config/datasets/tartanimu/lamar_full.yaml`**

For reference, the older lamar-v2-NO-GRAVITY-split dataset achieves val loss **0.1010** with identical hyperparameters on its own val set. Closing or beating that gap on the ios data is a stretch goal.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the key files** for full context:
   - `CLAUDE.md` — architecture and codebase overview
   - `model/model_lstm.py` — the FoundationModel (ResNet-LSTM with per-motion-type heads)
   - `model/function.py` — forward pass variants (train/test)
   - `model/losses.py` — multi-head masked loss
   - `train.py` — Trainer class with train loop, inference, checkpointing
   - `dataloader/dataset_AirLab.py` — data loading, windowing, augmentation
   - `main_net.py` — entry point, data path resolution, loader construction
   - `config/datasets/tartanimu/lamar_full.yaml` — the experiment config
   - `config/resnet_lstm_multihead.yaml` — model architecture config
4. **Verify data exists**: Check that `data/new_lamar_split/human/train/` contains 108 npz symlinks and `data/new_lamar_split/human/val/` contains 19 npz symlinks. Verify a symlink resolves: `python3 -c "import numpy as np; d=np.load('data/new_lamar_split/human/train/CAB_ios_2021-06-02_14.21.49.npz'); print('OK:', d['retargetted_imu'].shape)"`
5. **Create output dir**: `mkdir -p data/outputs/lamar_full`
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Data

The dataset is 127 iOS IMU recordings from the LaMAR dataset (200Hz, 6-channel: accel + gyro):
- **Train**: 108 files in `data/new_lamar_split/human/train/` (symlinks to npz files)
- **Val**: 19 files in `data/new_lamar_split/human/val/`
- **Test dir**: `data/new_lamar_split/human/test/` (empty, safely skipped by the code)

Each npz contains: `retargetted_ts`, `retargetted_imu` (N×6), `retargetted_pos` (N×3), `retargetted_quat` (N×4).

The model predicts body-frame velocity from IMU windows. Loss is MSE between predicted and ground-truth mean body velocity per window.

## Experimentation

Each experiment runs on a single GPU. Launch training as:

```bash
python main_net.py --config config/datasets/tartanimu/lamar_full.yaml > run.log 2>&1
```

**Time budget**: Use `epochs: 5` in the config for quick iteration (~10 min). Val loss converges by epoch 2-3, so 5 epochs is enough to evaluate an idea. Use `epochs: 20` only for final validation of promising changes.

**What you CAN modify** (these are the levers for improvement):
- `model/model_lstm.py` — model architecture (ResNet depth, LSTM size, attention, etc.)
- `model/function.py` — forward pass, how predictions are computed
- `model/losses.py` — loss functions
- `train.py` — training loop, optimizer, scheduler, mixed precision settings
- `dataloader/dataset_AirLab.py` — data loading, windowing, augmentation strategy
- `config/datasets/tartanimu/lamar_full.yaml` — hyperparameters (batch size, LR, augmentation, seq_len, etc.)
- `config/resnet_lstm_multihead.yaml` — model architecture params (layer_sizes, lstm_size, dropout, lstm_layers)
- `main_net.py` — data loading pipeline, DataLoader construction

**What you CANNOT modify**:
- The data files themselves (the npz files are read-only)
- The train/val split (must remain the same 108/19 files)
- Do NOT install new packages — only use what's in `requirement.txt`

**Dual objectives**: lowest `val_loss` (MSE) AND fastest `epoch_time`. A faster run that maintains val loss is a win. A lower val loss at the same speed is a win. Ideally both improve.

**Simplicity criterion**: All else equal, simpler is better. A tiny val_loss improvement from ugly complexity is not worth it. Removing code that gets equal results is a great outcome.

**The first run**: Always establish the baseline first — run with the config as-is and record the result.

## Extracting results

After each run, extract key metrics:

```bash
# Best val loss (last line, but check all — pick the lowest)
grep "val loss:" run.log

# Epoch timestamps (to compute epoch time)
grep "Training Epoch" run.log

# Check for crashes
tail -50 run.log
```

Val loss is logged as: `INFO     val loss: X.XXXXXX, val mse: X.XXXXXX`

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated).

Header and 6 columns:

```
commit	val_loss	epoch_time_s	epochs	status	description
```

1. git commit hash (short, 7 chars)
2. best val_loss achieved (e.g. 0.109000) — use 0.000000 for crashes
3. approximate epoch time in seconds (e.g. 120) — use 0 for crashes
4. number of epochs run
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_loss	epoch_time_s	epochs	status	description
a1b2c3d	0.109000	120	20	keep	baseline
b2c3d4e	0.107500	115	5	keep	increase LSTM hidden to 256
c3d4e5f	0.112000	95	5	discard	remove augmentation (faster but worse loss)
d4e5f6g	0.000000	0	0	crash	transformer encoder OOM
```

## Ideas to explore (non-exhaustive)

**Architecture**:
- Increase/decrease LSTM hidden size (currently 128) or layers (currently 2)
- Replace LSTM with GRU (faster, often comparable)
- Add attention mechanism on temporal features
- Modify ResNet1D backbone: change layer_sizes [2,2,2,2], try wider/shallower
- Add skip connections between ResNet and output head
- Try 1D depthwise-separable convolutions for speed

**Training**:
- Learning rate schedules: cosine annealing, warmup+cosine, OneCycleLR
- Different optimizers: AdamW with proper weight decay, LAMB
- Gradient clipping
- Adjust weight_decay (currently 0.01)
- Adjust LR (currently 0.0005)

**Data/Augmentation**:
- Tune augmentation noise levels (accel_bias_range=0.1, gyro_bias_range=0.002, gravity_noise_theta_range=5)
- Reduce or remove augmentation that hurts more than helps
- Change sample_freq (currently 40, controls step_size=5)
- Change seq_len (currently 10) — fewer steps per sample = faster
- Modify window_time in model config (currently 1.0s = 200 IMU frames per window)

**Speed**:
- Larger batch sizes (currently 128) — GPU may handle 256 or 512
- `torch.compile()` on the model
- Reduce n_workers or tune DataLoader settings
- Use channels_last memory format
- Profile and remove bottlenecks in the training loop
- Reduce logging frequency (currently every 100 batches)

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Make a change (architecture, hyperparams, training loop, etc.)
3. git commit
4. Run: `python main_net.py --config config/datasets/tartanimu/lamar_full.yaml > run.log 2>&1`
5. Read results: `grep "val loss:" run.log` and check epoch timestamps
6. If grep is empty, run crashed — `tail -n 50 run.log` for the traceback. Fix if easy.
7. Record in results.tsv (do NOT commit results.tsv — leave untracked)
8. If val_loss improved OR epoch_time improved without loss regression → keep the commit
9. If val_loss is worse AND no speed gain → `git reset --hard HEAD~1`

You are an autonomous researcher. Keep/discard based on results and iterate.

**Timeout**: Each 5-epoch run should take ~10 minutes. If a run exceeds 15 minutes, kill it (`pkill -f main_net.py`) and treat as failure.

**Crashes**: Fix typos and re-run. If the idea is fundamentally broken, log "crash", revert, move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. They may be away. Work indefinitely until manually stopped. If you run out of ideas, re-read the code, try combining near-misses, try radical changes. The loop runs until interrupted.
