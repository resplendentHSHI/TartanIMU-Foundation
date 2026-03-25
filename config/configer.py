# from model import transformer  # Not available
import logging
import os
import pdb
import re
import test

import torch
import torch.distributed as dist
import torch.nn as nn
import train
import yaml
from model import model_lstm
from termcolor import colored
from utils.rich_logging import (
    alert,
    attention_grabber,
    banner,
    celebration,
    critical,
    error,
    flash_message,
    highlight,
    info,
    info_highlight,
    print_config_summary,
    print_dataset_status,
    rich_logger,
    success,
    success_highlight,
    warning,
    warning_highlight,
)


# General config
def load_config(path):
    """Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    # Load configuration from file itself
    with open(path, "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    return cfg_special


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def build_model(args, cfg):
    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    )
    with open(cfg["model"]["model_yaml"], "r") as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)
    update_recursive(cfg, cfg_special)
    model_name = cfg["model"]["model_name"]  # resnet_lstm
    if model_name == "resnet_lstm":
        model = model_lstm.ResNetLSTMSeqNet(cfg)
    elif model_name == "resnet_lstm_light":
        model = model_lstm.ResNetLSTMSeqNet_Light(cfg)
    elif model_name == "IMU_transformer":
        # model = transformer.IMU_transformer()  # Not available
        raise NotImplementedError("IMU_transformer model is not available")
    elif model_name == "Foundation_Model":
        if cfg["model"].get("cross_xyz", False):
            lstm_trunk = model_lstm.Crossxy_LSTM_Model(cfg)
        else:
            lstm_trunk = model_lstm.ResNetLSTMSeqNet(cfg)
        dog_head = model_lstm.OutputHead(cfg, "dog")
        drone_head = model_lstm.OutputHead(cfg, "drone")
        car_head = model_lstm.OutputHead(cfg, "car")
        human_head = model_lstm.OutputHead(cfg, "human")
        heads = nn.ModuleDict(
            {"dog": dog_head, "human": human_head, "car": car_head, "drone": drone_head}
        )
        model = model_lstm.FoundationModel(cfg=cfg, trunk=lstm_trunk, heads=heads)

    ## use the multi GPU to train the network
    if cfg["train"]["use_multi_gpu"]:
        logging.info(f"torch.cuda.device_count() {torch.cuda.device_count()}")

        # Check if distributed process group is initialized
        if not dist.is_initialized():
            logging.warning(
                "Distributed process group not initialized. Falling back to single GPU mode."
            )
            network = model.to(device)
            total_params = network.get_num_params()
        else:
            # Note: torch.distributed.init_process_group() is already called in main_net.py
            torch.cuda.set_device(args.local_rank)
            
            # Check if distributed training is properly initialized before calling barrier
            if dist.is_initialized():
                dist.barrier()
            else:
                logging.warning("Distributed training not initialized, skipping barrier")
                
            # SyncBN
            model = (
                nn.SyncBatchNorm.convert_sync_batchnorm(model)
                .to(device)
                .cuda(args.local_rank)
            )
            # network = model.to(device)
            # total_params = network.get_num_params()
            network = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            # Check if network is wrapped in DistributedDataParallel
            if hasattr(network, "module"):
                total_params = network.module.get_num_params()
            else:
                total_params = network.get_num_params()
            
            # Check if distributed training is properly initialized before calling barrier
            if dist.is_initialized():
                dist.barrier()
            else:
                logging.warning("Distributed training not initialized, skipping barrier")
    else:
        # Single GPU mode - don't use distributed training
        logging.info("Running in single GPU mode - skipping distributed setup")
        network = model.to(device)
        total_params = network.get_num_params()  # 5358342

    # logging.info(f'Network "{model_name}" loaded to device {device}, device num {torch.cuda.device_count()}')
    # logging.info(f"Total number of parameters: {total_params}")

    return network


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split("([0-9]+)", v_str)]


def GetBestModel(path):
    # import pdb;pdb.set_trace()
    names = sorted(os.listdir(path + "/"), key=str2int)
    files = []
    for name in names:
        if os.path.isfile(os.path.join(os.path.abspath(path), name)):
            files.append(name)
    # files.sort()
    model = os.path.join(os.path.abspath(path), files[-1])  # Select the last model
    # model = os.path.join(os.path.abspath(path), 'checkpoint_197.pt') #Select the last model
    logging.info(f"load model: {model}")
    return model


def build_trainer(args, cfg, model, resume_path, **kwargs):
    # import pdb; pdb.set_trace()
    start_epoch = 0
    optim = (
        torch.optim.Adam
        if cfg["train"]["optimizer"]["method"] == "Adam"
        else torch.optim.SGD
    )  # Adam
    optimizer = optim(
        model.parameters(),
        cfg["train"]["optimizer"]["learning_rate"],  # 0.0001
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],
    )  # 0.0
    if cfg["train"]["use_pretrain_model"]:
        # Highlight model loading process
        banner("LOADING MODEL FOR FINETUNING")

        if resume_path:
            checkpoint = torch.load(resume_path)
            success_highlight(f"✅ LOADING MODEL FROM SPECIFIED PATH: {resume_path}")
        else:  # default resume form train out_dir
            resume_path = GetBestModel(
                os.path.join(cfg["train"]["out_dir"], "checkpoints")
            )
            checkpoint = torch.load(resume_path)
            success_highlight(f"✅ LOADING BEST MODEL FROM OUTPUT DIR: {resume_path}")

        # Get model information
        start_epoch = checkpoint.get("epoch", 0)
        model_epoch = checkpoint.get("epoch", "Unknown")

        # Create model information table
        from rich import box
        from rich.table import Table

        model_table = Table(title="📊 Model Loading Information", box=box.ROUNDED)
        model_table.add_column("Property", style="cyan", no_wrap=True)
        model_table.add_column("Value", style="green")
        model_table.add_column("Status", style="yellow")

        model_table.add_row("Starting Epoch", str(start_epoch), "✅")
        model_table.add_row("Model Epoch", str(model_epoch), "✅")
        model_table.add_row("Checkpoint Path", str(resume_path), "✅")
        model_table.add_row(
            "GPU Mode",
            "Multi-GPU" if cfg["train"]["use_multi_gpu"] else "Single-GPU",
            "✅",
        )

        # Load model weights
        if cfg["train"]["use_multi_gpu"]:
            # Check if model is wrapped in DistributedDataParallel
            if hasattr(model, "module"):
                model.module.load_state_dict(
                    checkpoint.get("model_state_dict"), strict=False
                )
                model_table.add_row("Model Weights", "Loaded (Multi-GPU)", "✅")
            else:
                model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
                model_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")
        else:
            model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
            model_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
            model_table.add_row("Optimizer State", "Loaded from checkpoint", "✅")
        else:
            model_table.add_row("Optimizer State", "Fresh optimizer (not found)", "⚠️")

        # Display the table
        from rich.console import Console

        console = Console()
        console.print(model_table)

        success_highlight(f"🎯 READY TO CONTINUE TRAINING FROM EPOCH {start_epoch}")
    else:
        # Create table for fresh training
        from rich import box
        from rich.table import Table

        fresh_table = Table(title="🆕 Fresh Training Information", box=box.ROUNDED)
        fresh_table.add_column("Property", style="cyan", no_wrap=True)
        fresh_table.add_column("Value", style="green")
        fresh_table.add_column("Status", style="yellow")

        fresh_table.add_row("Training Mode", "Fresh Training", "🆕")
        fresh_table.add_row("Starting Epoch", "0", "✅")
        fresh_table.add_row("Model State", "Random Initialization", "✅")
        fresh_table.add_row("Optimizer State", "Fresh Optimizer", "✅")
        fresh_table.add_row(
            "GPU Mode",
            "Multi-GPU" if cfg["train"]["use_multi_gpu"] else "Single-GPU",
            "✅",
        )

        from rich.console import Console

        console = Console()
        console.print(fresh_table)
    return train.Trainer(
        args, cfg, model=model, optimizer=optimizer, start_epoch=start_epoch
    )


def build_tester(args, cfg, model, check_path, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Highlight test model loading process
    banner("LOADING MODEL FOR TESTING")

    if check_path:
        checkpoint = torch.load(check_path, map_location=device)
        success_highlight(f"✅ LOADING TEST MODEL FROM SPECIFIED PATH: {check_path}")
    else:  # default resume form test out_dir
        checkpoint_path = GetBestModel(
            os.path.join(cfg["train"]["out_dir"], "checkpoints")
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        success_highlight(f"✅ LOADING BEST MODEL FOR TESTING: {checkpoint_path}")

    # Get model information
    model_epoch = checkpoint.get("epoch", "Unknown")

    # Create test model information table
    from rich import box
    from rich.table import Table

    test_table = Table(title="🧪 Test Model Loading Information", box=box.ROUNDED)
    test_table.add_column("Property", style="cyan", no_wrap=True)
    test_table.add_column("Value", style="green")
    test_table.add_column("Status", style="yellow")

    test_table.add_row("Model Epoch", str(model_epoch), "✅")
    test_table.add_row(
        "Checkpoint Path", str(check_path if check_path else checkpoint_path), "✅"
    )
    test_table.add_row(
        "GPU Mode", "Multi-GPU" if cfg["train"]["use_multi_gpu"] else "Single-GPU", "✅"
    )

    # Load model weights
    if cfg["train"]["use_multi_gpu"]:
        # Check if model is wrapped in DistributedDataParallel
        if hasattr(model, "module"):
            model.module.load_state_dict(
                checkpoint.get("model_state_dict"), strict=False
            )
            test_table.add_row("Model Weights", "Loaded (Multi-GPU)", "✅")
        else:
            model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
            test_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")
    else:
        model.load_state_dict(checkpoint.get("model_state_dict"), strict=False)
        test_table.add_row("Model Weights", "Loaded (Single-GPU)", "✅")

    model.eval()
    test_table.add_row("Model State", "Evaluation Mode", "✅")

    # Display the table
    from rich.console import Console

    console = Console()
    console.print(test_table)

    success_highlight("🎯 MODEL READY FOR TESTING")

    tester = test.tester(args, cfg, model)

    return tester
