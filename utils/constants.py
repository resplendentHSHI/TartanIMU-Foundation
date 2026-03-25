"""
Constants and configuration values for the neural inertial tracking system.
This file centralizes magic numbers and configuration values to improve maintainability.
"""

# Motion type mappings
MOTION_TYPES = {1: "car", 2: "dog", 3: "drone", 4: "human"}

MOTION_TYPE_NAMES = list(MOTION_TYPES.values())

# IMU data configuration
IMU_CHANNELS = 6  # [ax, ay, az, gx, gy, gz]
VELOCITY_DIM = 3  # [vx, vy, vz]
QUATERNION_DIM = 4  # [x, y, z, w]

# Default IMU frequency
DEFAULT_IMU_FREQ = 200  # Hz

# Physics constants
GRAVITY = 9.8101  # m/s^2

# Time window defaults
DEFAULT_WINDOW_TIME = 1.0  # seconds
DEFAULT_PAST_TIME = 0.0  # seconds
DEFAULT_FUTURE_TIME = 0.0  # seconds

# Model architecture defaults
DEFAULT_LSTM_SIZE = 128
DEFAULT_LSTM_LAYERS = 2
DEFAULT_LSTM_DROPOUT = 0.1
DEFAULT_DROP_RATIO = 0.3
DEFAULT_LAYER_SIZES = [2, 2, 2, 2]

# Training defaults
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 5

# Data loading defaults
DEFAULT_SEQ_LEN = 10
DEFAULT_N_WORKERS = 8
DEFAULT_TRAIN_RATE = 0.7
DEFAULT_VAL_RATE = 0.15
DEFAULT_TEST_RATE = 0.15

# File extensions
NPZ_EXTENSION = ".npz"
YAML_EXTENSION = ".yaml"
CHECKPOINT_EXTENSION = ".pt"

# Logging
LOG_FORMAT = "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# Distributed training
DEFAULT_MASTER_PORT = 12355
DEFAULT_NCCL_TIMEOUT = 1800

# Checkpoint saving
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.01  # 1% improvement
SAFETY_SAVE_INTERVAL = 3  # Save every N epochs as safety net
EMERGENCY_SAVE_THRESHOLD = 5  # Save if no save for N+ epochs

# Performance monitoring
MEMORY_TRACKING_INTERVAL = 100  # Track memory every N steps
TIMING_TRACKING_INTERVAL = 50  # Track timing every N steps

# Error handling
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRY_ATTEMPTS = 3

# Validation
MIN_SEQUENCE_LENGTH = 100  # Minimum sequence length for valid data
MAX_VELOCITY_THRESHOLD = 100.0  # Maximum reasonable velocity (m/s)
MIN_TIMESTAMP_DIFF = 1e-9  # Minimum timestamp difference (s)

# File paths
DEFAULT_OUTPUT_DIR = "/mnt/shibo_intern_project/result"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOG_DIR = "logs"

# Dataset names
DATASET_NAMES = {
    "TARTANAIR": "TarTanAir",
    "AIRLAB": "AirLab",
    "TRO": "TRO",
    "EUROC": "EuRoC",
    "KITTI": "KITTI",
}

# Model names
MODEL_NAMES = {
    "RESNET_LSTM": "resnet_lstm",
    "RESNET_LSTM_LIGHT": "resnet_lstm_light",
    "FOUNDATION_MODEL": "Foundation_Model",
    "IMU_TRANSFORMER": "IMU_transformer",
}

# Optimizer names
OPTIMIZER_NAMES = {"ADAM": "Adam", "SGD": "SGD", "ADAMW": "AdamW"}

# Scheduler names
SCHEDULER_NAMES = {
    "REDUCE_LR_ON_PLATEAU": "ReduceLROnPlateau",
    "MULTI_HEAD_REDUCE_LR": "MultiHeadReduceLROnPlateau",
    "COSINE_ANNEALING": "CosineAnnealingWarmRestarts",
}

# Optimizer defaults
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_MOMENTUM = 0.9
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999

# Scheduler defaults
DEFAULT_SCHEDULER = "ReduceLROnPlateau"
DEFAULT_SCHEDULER_FACTOR = 0.1
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_T0 = 10
DEFAULT_SCHEDULER_T_MULT = 2
DEFAULT_SCHEDULER_ETA_MIN = 1e-6

# Data loader defaults
DEFAULT_PIN_MEMORY = True
DEFAULT_PERSISTENT_WORKERS = True
DEFAULT_DROP_LAST = True
DEFAULT_SHUFFLE = True

# Model defaults
DEFAULT_MODEL_NAME = "Foundation_Model"
DEFAULT_PRED_VELOCITY = False
DEFAULT_USE_TRANSFORMER = False

# Training defaults
DEFAULT_USE_AMP = True
DEFAULT_USE_MULTI_GPU = False
DEFAULT_START_COV_EPOCHS = 40
DEFAULT_PREDICT_START = 0
DEFAULT_PREDICT_END = 10

# Logging defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Error handling defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 30.0

# Performance defaults
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_CACHE_SIZE = 1000
DEFAULT_MEMORY_FRACTION = 0.8

# File extensions
NPZ_EXTENSION = ".npz"
YAML_EXTENSION = ".yaml"
PYTHON_EXTENSION = ".py"

# Checkpoint and logging
CHECKPOINT_EXTENSION = ".pth"
LOG_EXTENSION = ".log"
TB_LOG_DIR = "logs"
WANDB_PROJECT = "neural-inertial-tracking"

# Error thresholds
EPSILON = 1e-6
MAX_GRAD_NORM = 1.0
MIN_LR = 1e-6

# Time constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
MILLISECONDS_PER_SECOND = 1000

# Data processing
MAX_SEQUENCE_LENGTH = 1000
MIN_SEQUENCE_LENGTH = 5
DEFAULT_SAMPLE_RATE = 200  # Hz
DEFAULT_WINDOW_OVERLAP = 0.5  # 50% overlap between windows

# Model architecture
MAX_LSTM_LAYERS = 4
MAX_LSTM_SIZE = 512
MAX_DROP_RATIO = 0.5
MIN_DROP_RATIO = 0.0

# Training
MAX_EPOCHS = 1000
MIN_EPOCHS = 1
MAX_BATCH_SIZE = 1024
MIN_BATCH_SIZE = 1
MAX_LEARNING_RATE = 1.0
MIN_LEARNING_RATE = 1e-8
MAX_PATIENCE = 100
MIN_PATIENCE = 1

# Validation
DEFAULT_VAL_FREQUENCY = 1  # Validate every N epochs
DEFAULT_SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
DEFAULT_LOG_FREQUENCY = 100  # Log every N steps

# Distributed training
DEFAULT_DIST_BACKEND = "nccl"
DEFAULT_DIST_URL = "env://"
DEFAULT_DIST_TIMEOUT = 1800  # 30 minutes

# Memory management
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_AMP_SCALER = 65536.0

# File paths
DEFAULT_OUTPUT_DIR = "./output"
DEFAULT_LOG_DIR = "./logs"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_CONFIG_DIR = "./config"
DEFAULT_DATA_DIR = "./data"

# Environment variables
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
WORLD_SIZE = "WORLD_SIZE"
RANK = "RANK"
LOCAL_RANK = "LOCAL_RANK"
MASTER_ADDR = "MASTER_ADDR"
MASTER_PORT = "MASTER_PORT"


# Utility functions
def torch_to_numpy(torch_data):
    """Convert torch tensor to numpy array."""
    return torch_data.cpu().detach().numpy()


def set_seed(seed):
    """Set random seed for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
