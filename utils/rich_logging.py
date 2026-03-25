"""
Rich logging utilities for beautiful terminal output.
Provides enhanced console output with colors, progress bars, and formatted tables.
"""

import sys
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()


class RichLogger:
    """Enhanced logger with rich formatting and colors."""

    def __init__(self, enable_rich: bool = True):
        self.enable_rich = enable_rich
        self.console = Console() if enable_rich else None

    def info(self, message: str, style: str = "blue"):
        """Log info message with rich formatting."""
        if self.enable_rich:
            self.console.print(f"[{style}]ℹ {message}[/{style}]")
        else:
            print(f"INFO: {message}")

    def success(self, message: str):
        """Log success message with green color."""
        if self.enable_rich:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(f"SUCCESS: {message}")

    def warning(self, message: str):
        """Log warning message with yellow color."""
        if self.enable_rich:
            self.console.print(f"[yellow]⚠️ {message}[/yellow]")
        else:
            print(f"WARNING: {message}")

    def error(self, message: str):
        """Log error message with red color."""
        if self.enable_rich:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"ERROR: {message}")

    def debug(self, message: str):
        """Log debug message with dim color."""
        if self.enable_rich:
            self.console.print(f"[dim]🐛 {message}[/dim]")
        else:
            print(f"DEBUG: {message}")

    def dataset_info(self, dataset_name: str, data_paths: Dict[str, str]):
        """Display dataset information in a formatted table."""
        if not self.enable_rich:
            print(f"Dataset: {dataset_name}")
            for key, path in data_paths.items():
                print(f"  {key}: {path}")
            return

        table = Table(title=f"📊 Dataset: {dataset_name}", box=box.ROUNDED)
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Path", style="green")
        table.add_column("Status", style="yellow")

        for data_type, path in data_paths.items():
            status = "✅" if path else "❌"
            table.add_row(data_type, str(path), status)

        self.console.print(table)

    def training_header(self, epoch: int, total_epochs: int, mode: str = "Training"):
        """Display training header with progress."""
        if not self.enable_rich:
            print(f"=== {mode} Epoch {epoch}/{total_epochs} ===")
            return

        progress = (epoch / total_epochs) * 100
        header = f"🚀 {mode} Epoch {epoch}/{total_epochs} ({progress:.1f}%)"
        self.console.print(Panel(header, style="bold blue", box=box.DOUBLE))

    def metrics_table(self, metrics: Dict[str, float], title: str = "Metrics"):
        """Display metrics in a formatted table."""
        if not self.enable_rich:
            print(f"{title}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            return

        table = Table(title=f"📈 {title}", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")

        for metric, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            table.add_row(metric, formatted_value)

        self.console.print(table)

    def checkpoint_info(self, epoch: int, reason: str, loss: float):
        """Display checkpoint saving information."""
        if not self.enable_rich:
            print(f"Checkpoint saved at epoch {epoch}: {reason} (loss: {loss:.6f})")
            return

        self.console.print(
            Panel(
                f"💾 Checkpoint saved\n"
                f"Epoch: {epoch}\n"
                f"Reason: {reason}\n"
                f"Loss: {loss:.6f}",
                title="Checkpoint",
                style="bold green",
                box=box.ROUNDED,
            )
        )

    def gpu_info(self, gpu_count: int, mode: str = "Multi-GPU"):
        """Display GPU configuration information."""
        if not self.enable_rich:
            print(f"{mode} training using {gpu_count} GPU(s)")
            return

        gpu_icon = "🖥️" if gpu_count == 1 else "🖥️" * min(gpu_count, 4)
        self.console.print(
            Panel(
                f"{gpu_icon} {mode} Training\n" f"Using {gpu_count} GPU(s)",
                title="Hardware Configuration",
                style="bold magenta",
                box=box.ROUNDED,
            )
        )

    def wandb_status(self, enabled: bool, rank: int = 0):
        """Display wandb logging status."""
        if not self.enable_rich:
            status = "enabled" if enabled else "disabled"
            print(f"Wandb logging: {status}")
            return

        if enabled and rank == 0:
            self.console.print("[green]📊 Wandb logging enabled (rank 0)[/green]")
        elif enabled and rank != 0:
            self.console.print("[dim]📊 Wandb disabled (rank {rank})[/dim]")
        else:
            self.console.print("[red]📊 Wandb logging disabled[/red]")

    def model_info(self, model_name: str, param_count: int):
        """Display model information."""
        if not self.enable_rich:
            print(f"Model: {model_name} ({param_count:,} parameters)")
            return

        self.console.print(
            Panel(
                f"🤖 Model: {model_name}\n" f"Parameters: {param_count:,}",
                title="Model Configuration",
                style="bold cyan",
                box=box.ROUNDED,
            )
        )


def create_progress_bar(description: str = "Training", total: Optional[int] = None):
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        description=description,
        total=total,
    )


def print_training_stats(
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    test_loss: Optional[float] = None,
    lr: Optional[float] = None,
    epoch_time: Optional[float] = None,
):
    """Print training statistics in a beautiful format."""
    if not console:
        # Fallback to simple print
        stats = f"Epoch {epoch}: Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            stats += f", Val Loss: {val_loss:.6f}"
        if test_loss is not None:
            stats += f", Test Loss: {test_loss:.6f}"
        if lr is not None:
            stats += f", LR: {lr:.6f}"
        if epoch_time is not None:
            stats += f", Time: {epoch_time:.2f}s"
        print(stats)
        return

    # Create rich table
    table = Table(title=f"📊 Training Statistics - Epoch {epoch}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    table.add_row("Train Loss", f"{train_loss:.6f}")
    if val_loss is not None:
        table.add_row("Val Loss", f"{val_loss:.6f}")
    if test_loss is not None:
        table.add_row("Test Loss", f"{test_loss:.6f}")
    if lr is not None:
        table.add_row("Learning Rate", f"{lr:.6f}")
    if epoch_time is not None:
        table.add_row("Epoch Time", f"{epoch_time:.2f}s")

    console.print(table)


def print_dataset_status(
    dataset_name: str, train_files: int, val_files: int, test_files: int
):
    """Print dataset status in a beautiful format."""
    if not console:
        print(
            f"Dataset: {dataset_name} - Train: {train_files}, Val: {val_files}, Test: {test_files}"
        )
        return

    table = Table(title=f"📁 Dataset: {dataset_name}", box=box.ROUNDED)
    table.add_column("Split", style="cyan", no_wrap=True)
    table.add_column("Files", style="green", justify="right")
    table.add_column("Status", style="yellow")

    table.add_row("Train", str(train_files), "✅" if train_files > 0 else "❌")
    table.add_row("Validation", str(val_files), "✅" if val_files > 0 else "❌")
    table.add_row("Test", str(test_files), "✅" if test_files > 0 else "❌")

    console.print(table)


def print_config_summary(config: Dict[str, Any]):
    """Print configuration summary in a beautiful format."""
    if not console:
        print("Configuration loaded successfully")
        return

    table = Table(title="⚙️ Configuration Summary", box=box.ROUNDED)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="blue")
    table.add_column("Value", style="green")

    for section, section_data in config.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                if isinstance(value, (int, float, str, bool)):
                    table.add_row(section, key, str(value))
                elif isinstance(value, list):
                    table.add_row(section, key, f"[{len(value)} items]")
                else:
                    table.add_row(section, key, str(type(value).__name__))
        else:
            table.add_row(section, "", str(section_data))

    console.print(table)


# Global logger instance
rich_logger = RichLogger()


# Convenience functions
def info(message: str, style: str = "blue"):
    rich_logger.info(message, style)


def success(message: str):
    rich_logger.success(message)


def warning(message: str):
    rich_logger.warning(message)


def error(message: str):
    rich_logger.error(message)


def debug(message: str):
    rich_logger.debug(message)


def highlight(message: str, style: str = "bold bright_yellow on bright_red"):
    """Display a highly visible highlighted message with distinct colors."""
    if not rich_logger.enable_rich:
        print(f"HIGHLIGHT: {message}")
        return

    rich_logger.console.print(f"[{style}]🔥 {message} 🔥[/{style}]")


def alert(message: str):
    """Display an alert message with bright red background and white text."""
    if not rich_logger.enable_rich:
        print(f"ALERT: {message}")
        return

    rich_logger.console.print(
        f"[bold white on bright_red]🚨 ALERT: {message} 🚨[/bold white on bright_red]"
    )


def success_highlight(message: str):
    """Display a success message with bright green background."""
    if not rich_logger.enable_rich:
        print(f"SUCCESS: {message}")
        return

    rich_logger.console.print(
        f"[bold white on bright_green]✅ SUCCESS: {message} ✅[/bold white on bright_green]"
    )


def warning_highlight(message: str):
    """Display a warning message with bright yellow background."""
    if not rich_logger.enable_rich:
        print(f"WARNING: {message}")
        return

    rich_logger.console.print(
        f"[bold black on bright_yellow]⚠️ WARNING: {message} ⚠️[/bold black on bright_yellow]"
    )


def info_highlight(message: str):
    """Display an info message with bright blue background."""
    if not rich_logger.enable_rich:
        print(f"INFO: {message}")
        return

    rich_logger.console.print(
        f"[bold white on bright_blue]ℹ️ INFO: {message} ℹ️[/bold white on bright_blue]"
    )


def critical(message: str):
    """Display a critical message with blinking red text."""
    if not rich_logger.enable_rich:
        print(f"CRITICAL: {message}")
        return

    rich_logger.console.print(
        f"[blink bold bright_red]💥 CRITICAL: {message} 💥[/blink bold bright_red]"
    )


def celebration(message: str):
    """Display a celebration message with rainbow colors."""
    if not rich_logger.enable_rich:
        print(f"CELEBRATION: {message}")
        return

    rich_logger.console.print(
        f"[bold bright_magenta]🎉 {message} 🎉[/bold bright_magenta]"
    )
    rich_logger.console.print(f"[bold bright_cyan]✨ {message} ✨[/bold bright_cyan]")
    rich_logger.console.print(f"[bold bright_green]🌟 {message} 🌟[/bold bright_green]")


def banner(message: str, style: str = "bold bright_yellow on bright_red"):
    """Display a banner message that's impossible to miss."""
    if not rich_logger.enable_rich:
        print(f"BANNER: {message}")
        return

    # Create a banner with borders
    border = "=" * (len(message) + 20)
    rich_logger.console.print(f"[{style}]{border}[/{style}]")
    rich_logger.console.print(
        f"[{style}]    🚨 {message} 🚨    [/bold bright_yellow on bright_red]"
    )
    rich_logger.console.print(f"[{style}]{border}[/{style}]")


def flash_message(message: str, color: str = "bright_red"):
    """Display a flashing message that alternates colors."""
    if not rich_logger.enable_rich:
        print(f"FLASH: {message}")
        return

    import time

    for i in range(3):  # Flash 3 times
        rich_logger.console.print(f"[bold {color}]⚡ {message} ⚡[/bold {color}]")
        time.sleep(0.5)
        rich_logger.console.print(
            f"[bold bright_yellow]⚡ {message} ⚡[/bold bright_yellow]"
        )
        time.sleep(0.5)


def attention_grabber(message: str):
    """Display a message designed to grab maximum attention."""
    if not rich_logger.enable_rich:
        print(f"ATTENTION: {message}")
        return

    # Multiple lines with different colors and effects
    rich_logger.console.print(
        f"[blink bold bright_red]🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥[/blink bold bright_red]"
    )
    rich_logger.console.print(
        f"[blink bold bright_red]🔥  ATTENTION REQUIRED  🔥[/blink bold bright_red]"
    )
    rich_logger.console.print(
        f"[blink bold bright_red]🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥[/blink bold bright_red]"
    )
    rich_logger.console.print(
        f"[bold white on bright_red]   {message}   [/bold white on bright_red]"
    )
    rich_logger.console.print(
        f"[blink bold bright_red]🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥[/blink bold bright_red]"
    )
