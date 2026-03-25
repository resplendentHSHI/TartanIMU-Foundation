"""
Error handling utilities for the neural inertial tracking system.
Provides custom exceptions, error recovery mechanisms, and logging utilities.
"""

import logging
import sys
import time
import traceback
from functools import wraps
from typing import Any, Callable, Optional

from utils.constants import DEFAULT_TIMEOUT, MAX_RETRY_ATTEMPTS


class NeuralInertialError(Exception):
    """Base exception for neural inertial tracking system."""

    pass


class ConfigurationError(NeuralInertialError):
    """Raised when there's an issue with configuration files or parameters."""

    pass


class DataLoadingError(NeuralInertialError):
    """Raised when there's an issue loading or processing data."""

    pass


class ModelError(NeuralInertialError):
    """Raised when there's an issue with model initialization or forward pass."""

    pass


class ModelCreationError(NeuralInertialError):
    """Raised when there's an issue creating or initializing a model."""

    pass


class TrainingError(NeuralInertialError):
    """Raised when there's an issue during training."""

    pass


class ValidationError(NeuralInertialError):
    """Raised when validation fails."""

    pass


class CheckpointError(NeuralInertialError):
    """Raised when there's an issue with checkpoint saving/loading."""

    pass


class DistributedTrainingError(NeuralInertialError):
    """Raised when there's an issue with distributed training."""

    pass


def handle_exceptions(
    error_type: type = Exception,
    max_retries: int = MAX_RETRY_ATTEMPTS,
    timeout: float = DEFAULT_TIMEOUT,
    logger: Optional[logging.Logger] = None,
):
    """
    Decorator to handle exceptions with retry logic and timeout.

    Args:
        error_type: Type of exception to catch
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        logger: Logger instance for error logging

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)

                    # Check timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Function {func.__name__} exceeded timeout of {timeout}s"
                        )

                    return result

                except error_type as e:
                    last_exception = e

                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}"
                        )

                    if attempt < max_retries:
                        # Exponential backoff
                        wait_time = 2**attempt
                        time.sleep(wait_time)
                    else:
                        if logger:
                            logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}. "
                                f"Last error: {e}"
                            )
                        raise last_exception

                except Exception as e:
                    # Re-raise unexpected exceptions immediately
                    if logger:
                        logger.error(f"Unexpected error in {func.__name__}: {e}")
                    raise

            return None  # Should never reach here

        return wrapper

    return decorator


def safe_execute(
    func: Callable,
    *args,
    error_msg: str = "Function execution failed",
    logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        error_msg: Error message to log
        logger: Logger instance
        **kwargs: Function keyword arguments

    Returns:
        Function result or None if failed

    Raises:
        Exception: Re-raises the original exception
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"{error_msg}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        raise


def validate_input(
    value: Any,
    expected_type: type,
    name: str,
    allow_none: bool = False,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> None:
    """
    Validate input parameters.

    Args:
        value: Value to validate
        expected_type: Expected type
        name: Parameter name for error messages
        allow_none: Whether None values are allowed
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if not allow_none:
            raise ValidationError(f"{name} cannot be None")
        return

    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
        )

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")


def setup_error_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    include_traceback: bool = True,
) -> logging.Logger:
    """
    Setup error logging configuration.

    Args:
        log_file: Path to log file
        log_level: Logging level
        include_traceback: Whether to include traceback in error logs

    Returns:
        Configured logger
    """
    logger = logging.getLogger("neural_inertial_error")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] %(name)s:%(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ErrorRecovery:
    """Error recovery utilities."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.recovery_strategies = {}

    def register_recovery_strategy(self, error_type: type, strategy: Callable) -> None:
        """
        Register a recovery strategy for a specific error type.

        Args:
            error_type: Type of exception to handle
            strategy: Recovery function to call
        """
        self.recovery_strategies[error_type] = strategy

    def attempt_recovery(
        self, error: Exception, context: Optional[dict] = None
    ) -> bool:
        """
        Attempt to recover from an error using registered strategies.

        Args:
            error: The exception that occurred
            context: Additional context for recovery

        Returns:
            True if recovery was successful, False otherwise
        """
        error_type = type(error)

        if error_type in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error_type]
                strategy(error, context)
                self.logger.info(f"Successfully recovered from {error_type.__name__}")
                return True
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery strategy failed for {error_type.__name__}: {recovery_error}"
                )
                return False
        else:
            self.logger.warning(
                f"No recovery strategy registered for {error_type.__name__}"
            )
            return False


# Global error recovery instance
error_recovery = ErrorRecovery()


def log_error_with_context(
    error: Exception, context: dict, logger: Optional[logging.Logger] = None
) -> None:
    """
    Log an error with additional context information.

    Args:
        error: The exception that occurred
        context: Additional context information
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.error(f"Error: {error}")
    logger.error(f"Context: {context}")
    logger.error(f"Traceback: {traceback.format_exc()}")


def cleanup_on_error(
    cleanup_func: Callable, *cleanup_args, **cleanup_kwargs
) -> Callable:
    """
    Decorator to ensure cleanup function is called on error.

    Args:
        cleanup_func: Function to call for cleanup
        *cleanup_args: Arguments for cleanup function
        **cleanup_kwargs: Keyword arguments for cleanup function

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                try:
                    cleanup_func(*cleanup_args, **cleanup_kwargs)
                except Exception as cleanup_error:
                    logging.error(f"Cleanup failed: {cleanup_error}")
                raise e

        return wrapper

    return decorator
