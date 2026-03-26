# autoresearch — TartanIMU Foundation Model Training

This is an experiment to have the LLM autonomously train and optimize the TartanIMU foundation model.

## Objective

**Primary goal**: Train the full foundation model (all 4 motion types: car, dog, drone, human) on verified data, then incorporate LaMAR data, then hill-climb to minimize loss.

### Phase 1 — Train on Verified Data
Train the foundation model on the verified dataset (`verified_data/`) which contains all 4 motion types:
- **car**: 67 train / 16 val / 9 test
- **dog**: 41 train / 17 val / 14 test
- **drone**: 186 train / 30 val / 30 test
- **human**: 43 train / 14 val / 15 test

Goal: achieve reasonable train/val loss convergence across all heads. "Reasonable" means the model is learning (loss decreasing steadily, val loss tracking train loss without large divergence).

### Phase 2 — Combine LaMAR Data
Once Phase 1 loss is appropriate, combine the LaMAR human data (`data/new_lamar_split/human/`, 105 train / 19 val) into the training set alongside the verified data. Use `resume_from` to continue from the Phase 1 checkpoint. Monitor that the additional data helps rather than hurts — val loss should not regress significantly.

### Phase 3 — Hill Climbing
With all data combined, iterate on **everything** to reduce loss as far as possible. This includes hyperparameters, training strategy, **and model architecture** (layer sizes, LSTM hidden dims, number of layers, dropout, ResNet block configuration, etc.). Keep/discard each experiment based on whether it improves val loss.

### Data Split
Use a **75/25 train/val split** for all data. If the existing splits don't match this ratio, re-split using `random_partition: True` with `train_rate: 0.75` and `valid_rate: 0.25` in the config.

### Reproducible Split
The random partition is seeded by `seeds.id` in the config (default: 42). With `random_partition: True`, `np.random.seed` is set before `partition_data()` shuffles the index map, so the same seed always produces the same split. The partition function already logs which trajectories land in train vs val via `[DATA_SPLIT]` log lines.

**You must preserve this information**: After each Phase 1 run, extract the split from the log and save it to `data_split_log.txt`:

```bash
grep "\[DATA_SPLIT\]" run.log > data_split_log.txt
```

This file records the exact train/val assignment for every trajectory, making the split fully reconstructable. If you change the seed or the data composition, re-extract and overwrite `data_split_log.txt`. Keep this file in the repo (committed) so the split is always recoverable.

## Repository

**Remote**: `git@github.com:resplendentHSHI/TartanIMU-Foundation.git` (origin)

**Push at phase completion**: When a phase is complete (stable, good results), push the branch to origin as a record:
```bash
git push origin <branch-name>
```

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b <branch-name>` from current branch.
3. **Read the key files** for full context:
   - `CLAUDE.md` — architecture and codebase overview
   - `dataloader/dataset_AirLab.py` — data loading, windowing, augmentation
   - `model/model_lstm.py` — the FoundationModel (ResNet-LSTM with per-motion-type heads)
   - `model/function.py` — forward pass variants (train/test)
   - `model/losses.py` — multi-head masked loss
   - `train.py` — Trainer class with train loop, inference, checkpointing
   - `main_net.py` — entry point, data path resolution, loader construction
   - `config/datasets/tartanimu/foundation.yaml` — foundation model config (reference)
   - `config/resnet_lstm_multihead.yaml` — model architecture config
4. **Verify data exists**:
   - `verified_data/car/` — car motion data
   - `verified_data/dog/` — dog motion data
   - `verified_data/drone/` — drone motion data
   - `verified_data/human/` — human motion data
   - `data/new_lamar_split/human/` — LaMAR human data (for Phase 2)
5. **Create output dir**: `mkdir -p data/outputs/foundation_training`
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Create/update config**: Set up the training config for Phase 1 (verified data only, all 4 heads, 75/25 split).
8. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Data

### Verified Data (`verified_data/`)
Pre-verified IMU recordings across 4 motion types. Each motion type has its own subdirectory structure with `train/`, `val/`, and `test/` splits. The data has already been validated and is known-good.

### LaMAR Data (`data/new_lamar_split/human/`)
iOS IMU recordings from the LaMAR dataset (200Hz, 6-channel: accel + gyro). 105 train / 19 val npz files. This data was reprocessed and should be combined in Phase 2.

Each npz contains: `retargetted_ts`, `retargetted_imu` (N x 6), `retargetted_pos` (N x 3), `retargetted_quat` (N x 4).

The model predicts body-frame velocity from IMU windows. Loss is MSE between predicted and ground-truth mean body velocity per window.

## Experimentation

Each experiment runs on a single GPU. Launch training as:

```bash
WANDB_RUN_NAME="description-of-experiment" python main_net.py --config <config.yaml> > run.log 2>&1
```

**WandB labeling**: Always set `WANDB_RUN_NAME` to a descriptive label for each run (e.g. `phase1-verified-all-heads`, `phase2-add-lamar`, `phase3-lr-sweep`). This is read by `main_net.py` via `os.environ.get("WANDB_RUN_NAME", ...)`.

**Resume from checkpoint**: Use `--resume_from /path/to/checkpoint.pt` when transitioning between phases.

**Time budget**: Use `epochs: 5` for quick iteration. Use `epochs: 15-30` for phase transitions and serious evaluation.

**Priority of changes** (most to least impactful):
1. **Get all heads training on verified data** — Phase 1 baseline
2. **Combine LaMAR data** — Phase 2, expand human data
3. **Training hyperparameters** — LR, batch size, scheduler, optimizer, augmentation
4. **Architecture tweaks** — only if hyperparams are tuned and gains are plateauing

**What you SHOULD modify** (in priority order):
1. Training config YAML — data paths, split ratios, hyperparameters
2. Data split scripts if needed to achieve 75/25 split

**What you CANNOT modify (Phases 1 & 2)**:
- The raw/verified data files themselves
- Do NOT install new packages — only use what's in `requirement.txt`
- The model code (`model/model_lstm.py`, `model/function.py`, `model/losses.py`, `config/resnet_lstm_multihead.yaml`) — the model is published and assumed correct for Phases 1 & 2
- The data loading pipeline (`dataloader/dataset_AirLab.py`, `main_net.py`, `train.py`) — these are part of the published model code. Only modify if there is a clear bug.

**Phase 3 unlocks**: In Phase 3 (hill climbing), you MAY modify model architecture code and configs — layer sizes, LSTM hidden dimensions, number of layers, dropout rates, ResNet block counts, output head structure, `config/resnet_lstm_multihead.yaml`, etc. The goal is to minimize loss by any means. The data loading pipeline and data files remain off-limits.

**Simplicity criterion**: All else equal, simpler is better. A tiny val_loss improvement from ugly complexity is not worth it.

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

Header and 7 columns:

```
phase	commit	val_loss	epoch_time_s	epochs	status	description
```

1. phase: `1`, `2`, or `3`
2. git commit hash (short, 7 chars)
3. best val_loss achieved (e.g. 0.109000) — use 0.000000 for crashes
4. approximate epoch time in seconds (e.g. 120) — use 0 for crashes
5. number of epochs run
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
phase	commit	val_loss	epoch_time_s	epochs	status	description
1	a1b2c3d	0.450000	120	20	keep	baseline verified data all heads
1	b2c3d4e	0.380000	115	20	keep	increased batch size to 256
2	c3d4e5f	0.320000	130	15	keep	added lamar data, resumed from phase1 best
3	d4e5f6g	0.290000	125	15	keep	cosine annealing LR schedule
3	e5f6g7h	0.000000	0	0	crash	transformer encoder OOM
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Determine current phase:
   - **Phase 1**: Training on verified data only. Move to Phase 2 when val loss is converging and stable.
   - **Phase 2**: Add LaMAR data, resume from best Phase 1 checkpoint. Move to Phase 3 when the combined model is stable.
   - **Phase 3**: Hill climbing — try hyperparameter changes, augmentation tuning, etc.
3. Make a change (config, hyperparams, data combination, etc.)
4. git commit
5. Run: `WANDB_RUN_NAME="description" python main_net.py --config <config.yaml> > run.log 2>&1`
6. Read results: `grep "val loss:" run.log` and check epoch timestamps
7. If grep is empty, run crashed — `tail -n 50 run.log` for the traceback. Fix if easy.
8. Record in results.tsv (do NOT commit results.tsv — leave untracked)
9. If val_loss improved OR epoch_time improved without loss regression -> keep the commit
10. If val_loss is worse AND no speed gain -> `git reset --hard HEAD~1`

You are an autonomous researcher. Keep/discard based on results and iterate.

**Timeout**: Each 5-epoch run should take ~10 minutes. If a run exceeds 15 minutes, kill it (`pkill -f main_net.py`) and treat as failure.

**Crashes**: Fix typos and re-run. If the idea is fundamentally broken, log "crash", revert, move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. They may be away. Work indefinitely until manually stopped. If you run out of ideas, re-read the code, try combining near-misses, try radical changes. The loop runs until interrupted.

## Ideas to explore (non-exhaustive)

**Phase 1 — Verified Data Training**:
- Ensure all 4 heads (car, dog, drone, human) are active and training
- Verify the 75/25 split is applied correctly via `random_partition: True`
- Establish baseline loss per motion type
- Tune initial learning rate and batch size for multi-head training

**Phase 2 — LaMAR Integration**:
- Add LaMAR human data path alongside verified human data
- Resume from best Phase 1 checkpoint
- Monitor whether human head loss improves with more data
- Check that other heads don't regress

**Phase 3 — Hill Climbing (everything is fair game)**:
- Learning rate schedules: cosine annealing, warmup+cosine, OneCycleLR
- Different optimizers: AdamW with proper weight decay
- Gradient clipping
- Adjust weight_decay, LR, batch size
- Augmentation noise levels
- Sequence length tuning
- Covariance training activation timing
- **Model architecture changes**:
  - LSTM hidden size (e.g. 128 -> 256 -> 512)
  - Number of LSTM layers
  - ResNet block count and channel widths
  - Dropout rates
  - Output head layer sizes
  - Any structural change to `model/model_lstm.py` or `config/resnet_lstm_multihead.yaml`
