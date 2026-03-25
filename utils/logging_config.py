"""
Centralized logging configuration for the neural inertial tracking system.
Provides consistent logging setup across all modules.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from utils.constants import DEFAULT_LOG_DIR, LOG_FORMAT, LOG_LEVEL

from rich.columns import Columns
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

console = Console()


def console_log(message):
    console.log(message)


def print_columns(data, title=None):
    if title:
        console.print(title)
    console.print(Columns(data))



class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class TrainingLogger:
    """Specialized logger for training operations."""

    def __init__(
        self,
        name: str = "training",
        log_dir: Optional[str] = None,
        level: str = LOG_LEVEL,
        use_colors: bool = True,
    ):
        self.name = name
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.level = level
        self.use_colors = use_colors

        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup the logger with appropriate handlers and formatters."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.use_colors:
            console_formatter = ColoredFormatter(LOG_FORMAT)
        else:
            console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(self.log_dir, f"{self.name}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

    def log_training_step(
        self, epoch: int, step: int, loss: float, learning_rate: float, **kwargs
    ) -> None:
        """Log training step information."""
        self.logger.info(
            f"Epoch {epoch}, Step {step}: Loss={loss:.6f}, LR={learning_rate:.6f}"
        )

        # Log additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_validation(
        self, epoch: int, val_loss: float, metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log validation results."""
        self.logger.info(f"Epoch {epoch} Validation: Loss={val_loss:.6f}")

        if metrics:
            for metric_name, metric_value in metrics.items():
                self.logger.info(f"  {metric_name}: {metric_value:.6f}")

    def log_checkpoint(self, epoch: int, path: str, is_best: bool = False) -> None:
        """Log checkpoint saving."""
        status = "BEST" if is_best else "Regular"
        self.logger.info(f"Epoch {epoch}: Saved {status} checkpoint to {path}")

    def log_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error with context."""
        self.logger.error(f"Error: {error}")
        if context:
            self.logger.error(f"Context: {context}")
        self.logger.error(f"Traceback: {error.__traceback__}")


class PerformanceLogger:
    """Specialized logger for performance monitoring."""

    def __init__(
        self,
        name: str = "performance",
        log_dir: Optional[str] = None,
        level: str = "INFO",
    ):
        self.name = name
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.level = level

        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup the performance logger."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper()))

        # Clear existing handlers
        logger.handlers.clear()

        # File handler only for performance logs
        log_file = os.path.join(self.log_dir, f"{self.name}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def log_memory_usage(
        self,
        gpu_memory_allocated: float,
        gpu_memory_reserved: float,
        cpu_memory_percent: float,
        context: str = "",
    ) -> None:
        """Log memory usage information."""
        self.logger.info(
            f"Memory Usage {context}: "
            f"GPU Allocated={gpu_memory_allocated:.2f}MB, "
            f"GPU Reserved={gpu_memory_reserved:.2f}MB, "
            f"CPU={cpu_memory_percent:.1f}%"
        )

    def log_timing(self, operation: str, duration: float, context: str = "") -> None:
        """Log timing information."""
        self.logger.info(f"Timing {context}: {operation} took {duration:.4f}s")

    def log_throughput(self, samples_per_second: float, context: str = "") -> None:
        """Log throughput information."""
        self.logger.info(f"Throughput {context}: {samples_per_second:.2f} samples/sec")


def setup_logging(
    name: str = "neural_inertial",
    log_dir: Optional[str] = None,
    level: str = LOG_LEVEL,
    use_colors: bool = True,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Setup centralized logging configuration.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        use_colors: Whether to use colored output
        log_to_file: Whether to log to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if use_colors:
        console_formatter = ColoredFormatter(LOG_FORMAT)
    else:
        console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = log_dir or DEFAULT_LOG_DIR
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_to_wandb(header: str, data: Any, epoch: int):
    """Log data to Weights & Biases."""
    try:
        import wandb

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"{header}/{key}": value}, step=epoch)
        else:
            wandb.log({header: data}, step=epoch)
    except ImportError:
        logger = get_logger("wandb")
        logger.warning("wandb not installed, skipping logging")


def log_function_call(func_name: str, args: tuple, kwargs: dict) -> None:
    """
    Log function call information for debugging.

    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    logger = get_logger("function_calls")
    logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")


def log_function_result(func_name: str, result: Any, duration: float) -> None:
    """
    Log function result and timing.

    Args:
        func_name: Name of the function
        result: Function result
        duration: Function execution time
    """
    logger = get_logger("function_results")
    logger.debug(f"{func_name} returned {result} in {duration:.4f}s")


def setup_experiment_logging(
    experiment_name: str, log_dir: Optional[str] = None
) -> Dict[str, logging.Logger]:
    """
    Setup logging for a specific experiment.

    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files

    Returns:
        Dictionary of loggers for different components
    """
    log_dir = log_dir or os.path.join(DEFAULT_LOG_DIR, experiment_name)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    loggers = {}

    # Main experiment logger
    loggers["main"] = setup_logging(
        f"{experiment_name}_main", log_dir=log_dir, use_colors=True, log_to_file=True
    )

    # Training logger
    loggers["training"] = TrainingLogger(
        name=f"{experiment_name}_training", log_dir=log_dir
    ).logger

    # Performance logger
    loggers["performance"] = PerformanceLogger(
        name=f"{experiment_name}_performance", log_dir=log_dir
    ).logger

    # Error logger
    loggers["error"] = setup_logging(
        f"{experiment_name}_error", log_dir=log_dir, use_colors=False, log_to_file=True
    )

    return loggers
