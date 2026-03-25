#!/usr/bin/env python3
"""
Fine-tune Example for Neural Inertial Tracking Model
===================================================

This script provides a clean example for:
1. Loading a pre-trained model
2. Running inference on real IMU data
3. Fine-tuning the covariance head
4. Evaluating model performance using existing evaluation functions

This version reuses existing code from the codebase for data loading, evaluation, and training.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import datetime
import json
import time
import os.path as osp

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing code from the codebase
from model.model_lstm import FoundationModel, ResNetLSTMSeqNet, OutputHead
from model import function
from model.losses import single_head_mask_loss, smooth_transition_weight
from config.configer import load_config, build_model
from dataloader.dataset_AirLab import ResNetLSTMSeqToSeqDataset, BasicSequenceData
from evaluation.metrics import compute_accruacy_metrics, compute_ate_rte
from evaluation import postprocess
from torch.utils.data import DataLoader
from utils.rich_logging import info, success, warning, error, banner

# Import functions from test.py to reuse existing evaluation code
import test
# Reuse all evaluation functions from test.py
compute_overall_statistics = test.compute_overall_statistics
display_rich_metrics_tables = test.display_rich_metrics_tables
save_comprehensive_csv_files = test.save_comprehensive_csv_files
create_segments_summary_plot = test.create_segments_summary_plot
segment_trajectory_5m = test.segment_trajectory_5m
process_trajectory_segments = test.process_trajectory_segments
compute_aggregated_metrics = test.compute_aggregated_metrics
compute_full_trajectory_metrics = test.compute_full_trajectory_metrics
save_metrics_files = test.save_metrics_files
create_output_directories = test.create_output_directories
create_test_dataloader = test.create_test_dataloader


class FineTuneInertialTracker:
    """
    Fine-tune wrapper for neural inertial tracking model.

    """
    
    def __init__(self, config_path: str, model_path: str, device: str = "cuda"):
        """
        Initialize the tracker.
        
        Args:
            config_path: Path to model configuration YAML file
            model_path: Path to pre-trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.model_path = model_path
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration using existing config system
        self.config = self._load_config()
        
        # Initialize model using existing model building system
        self.model = self._load_model()
        
        info(f"FineTuneInertialTracker initialized on device: {self.device}")
        info(f"Model parameters: {self.model.get_num_params():,}")
    
    def _load_config(self) -> Dict:
        """Load model configuration using existing config system."""
        info(f"Loading configuration from: {self.config_path}")
        config = load_config(self.config_path)
        info(f"Configuration loaded successfully")
        return config
    
    def _load_model(self) -> FoundationModel:
        """Load and initialize the model using existing model building system."""
        # Create a dummy args object for the build_model function
        class DummyArgs:
            def __init__(self):
                self.local_rank = 0
        
        args = DummyArgs()
        
        # Use existing model building system
        model = build_model(args, self.config)
        
        # Load pre-trained weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            info(f"Loaded pre-trained model from: {self.model_path}")
        else:
            warning(f"Model path not found: {self.model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('finetune_example.log')
            ]
        )
    
    def load_real_data(self, data_path: str, mode: str = "test") -> DataLoader:
        """
        Load real IMU data using existing data loading system.
        
        Args:
            data_path: Path to the .npz data file
            mode: Data mode ('train', 'val', 'test')
            
        Returns:
            DataLoader for the specified data
        """
        info(f"Loading real data from: {data_path}")
        
        # Use existing data loading system
        from dataloader import dataset_AirLab as dataset_utils
        
        # Create basic data object
        basic_data = dataset_utils.BasicSequenceData(self.config, [data_path], mode=mode)
        
        # Create dataset using existing dataset class
        dataset = dataset_utils.ResNetLSTMSeqToSeqDataset(
            self.config, basic_data, basic_data.get_merged_index_map(), mode=mode
        )
        
        # Create dataloader
        batch_size = self.config.get("test", {}).get("batch_size", 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))
        
        info(f"Created dataloader with {len(dataset)} samples, batch_size={batch_size}")
        return dataloader
    
    def inference_on_real_data(self, data_path: str, motion_type: str = "drone", 
                              predict_covariance: bool = False) -> Dict:
        """
        Run inference on real IMU data.
        
        Args:
            data_path: Path to the .npz data file
            motion_type: Motion type ('car', 'dog', 'drone', 'human')
            predict_covariance: Whether to predict covariance
            
        Returns:
            Dictionary containing predictions and optionally covariance
        """
        info(f"Running inference on real data: {data_path}")
        
        # Load data
        dataloader = self.load_real_data(data_path, mode="test")
        
        self.model.eval()
        all_predictions = []
        all_covariances = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # The dataset returns (seq_feat, targ, ori, label) tuple
                imu_data = batch[0].to(self.device)  # seq_feat
                targets = batch[1].to(self.device)   # targ
                motion_types = batch[3].to(self.device)  # label
                
                # Run inference
                if predict_covariance:
                    predictions, covariances = self.model(
                        imu_data, 
                        motion_type=motion_types,
                        predict_cov=True,
                        compute_all_heads=False
                    )
                    all_covariances.append(covariances[motion_type].cpu().numpy())
                else:
                    predictions = self.model(
                        imu_data,
                        motion_type=motion_types,
                        predict_cov=False,
                        compute_all_heads=False
                    )
                
                all_predictions.append(predictions[motion_type].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    info(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate all results
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        result = {
            'predictions': predictions,
            'targets': targets,
            'motion_type': motion_type,
            'data_path': data_path
        }
        
        if predict_covariance:
            result['covariances'] = np.concatenate(all_covariances, axis=0)
        
        info(f"Inference completed. Predictions shape: {predictions.shape}")
        return result
    
    def fine_tune_covariance_head(self, train_data_path: str, 
                                 motion_type: str = "drone",
                                 epochs: int = 10,
                                 learning_rate: float = 1e-4,
                                 debug_mode: bool = False) -> Dict:
        """
        Fine-tune the covariance head using real data with consistent loss function.
        
        Args:
            train_data_path: Path to training data
            motion_type: Motion type to fine-tune for
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Training history dictionary
        """
        info(f"Starting covariance head fine-tuning for {motion_type}")
        
        # Load training data using existing system
        train_loader = self.load_real_data(train_data_path, mode="train")
        
        # Setup optimizer for covariance head training
        if debug_mode:
            # Debug mode: Freeze velocity head and only train covariance head
            info("DEBUG MODE: Freezing velocity head, training only covariance head")
            
            # Freeze velocity head parameters (output_block1) and keep covariance head trainable
            for name, param in self.model.named_parameters():
                if f"heads.{motion_type}.output_block1" in name:
                    param.requires_grad = False
                    info(f"🔒 Frozen velocity parameter: {name}")
                elif f"heads.{motion_type}.output_block2" in name:
                    param.requires_grad = True
                    info(f"🔄 Training covariance parameter: {name}")
                else:
                    # Freeze all other parameters to preserve pre-trained performance
                    param.requires_grad = False
                    info(f"🔒 Frozen parameter: {name}")
            
            # Only train covariance head parameters
            trainable_params = [param for param in self.model.parameters() if param.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
            info(f"Debug mode: Training {len(trainable_params)} parameters (only covariance head)")
        else:
            # Normal mode: Train covariance head
            info(" NORMAL MODE: Training covariance head")
            
            # We need to ensure all parameters are trainable during forward pass
            # but only update covariance head parameters
            covariance_params = []
            all_params = list(self.model.parameters())
            
            # First, make all parameters trainable for the forward pass
            for param in all_params:
                param.requires_grad = True
            
            # Then identify covariance head parameters for the optimizer
            for name, param in self.model.named_parameters():
                if f"heads.{motion_type}.output_block2" in name:
                    covariance_params.append(param)
                    info(f"Training parameter: {name}")
            
            if not covariance_params:
                warning(f"No covariance parameters found for motion_type: {motion_type}")
                # Fallback: train all parameters
                covariance_params = all_params
                info("Falling back to training all parameters")
            else:
                info(f"Found {len(covariance_params)} covariance parameters to train")
            
            optimizer = torch.optim.Adam(covariance_params, lr=learning_rate)
        
        # Training loop
        self.model.train()
        info(f"Model training mode: {self.model.training}")
        history = {'loss': [], 'grad_norm': [], 'param_norm': []}
        
        # Store initial model state for comparison
        initial_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                initial_state[name] = param.data.clone()
        
        # Get configuration parameters
        start_cov_epochs = self.config.get("train", {}).get("start_cov_epochs", 0)

        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # The dataset returns (seq_feat, targ, ori, label) tuple
                imu_data = batch[0].to(self.device)  # seq_feat
                targets = batch[1].to(self.device)   # targ
                motion_types = batch[3].to(self.device)  # label
                
                # Ensure model is in training mode for gradient computation
                self.model.train()
                
                # Forward pass with covariance prediction
                predictions, covariances = self.model(
                    imu_data,
                    motion_type=motion_types,
                    predict_cov=True,
                    compute_all_heads=False
                )
                
                # Create mask for the current motion type
                motion_type_mask = motion_types == self._get_motion_type_id(motion_type)
                
                # Use the same training function as the main training code for consistency
                # This ensures we use the exact same loss function and forward pass
                batch_for_training = [imu_data, targets, torch.zeros_like(targets), motion_types]
                
                # Temporarily enable gradients for all parameters during forward pass
                # This ensures the loss computation is properly connected
                temp_grad_states = {}
                for name, param in self.model.named_parameters():
                    temp_grad_states[name] = param.requires_grad
                    param.requires_grad = True
                
                # For overfitting experiment, use simple MSE loss to avoid coupling issues
                if debug_mode:
                    # Simple MSE loss on covariance predictions for overfitting test
                    cov_pred = covariances[motion_type]
                    # Use a simple target for covariance (e.g., constant uncertainty)
                    cov_target = torch.ones_like(cov_pred) * 0.1  # Target uncertainty
                    loss = torch.nn.functional.mse_loss(cov_pred, cov_target)
                else:
                    # Use the efficient training forward function (same as train.py)
                    pred, pred_cov, targ, loss = function.fun_train_forward_efficient(
                        self.config, self.model, batch_for_training, start_cov_epochs, epoch + 1
                    )
                
                # Restore original gradient states
                for name, param in self.model.named_parameters():
                    param.requires_grad = temp_grad_states[name]
                
                # Log training progress with gradient monitoring
                if batch_idx % 10 == 0:
                    # Calculate gradient norm
                    grad_norm = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    
                    # Calculate parameter norm
                    param_norm = 0.0
                    for param in self.model.parameters():
                        if param.requires_grad:
                            param_norm += param.data.norm(2).item() ** 2
                    param_norm = param_norm ** 0.5
                    
                    info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.6f}, Param Norm: {param_norm:.6f}")
                    
                    # Check for gradient explosion
                    if grad_norm > 10.0:
                        warning(f" Large gradient detected: {grad_norm:.6f}")
                    
                    history['grad_norm'].append(grad_norm)
                    history['param_norm'].append(param_norm)
                
                # Backward pass with gradient clipping for stability
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            
            # Calculate parameter change
            total_param_change = 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in initial_state:
                    param_change = (param.data - initial_state[name]).norm(2).item()
                    total_param_change += param_change ** 2
            total_param_change = total_param_change ** 0.5
            
            info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}, Total Param Change: {total_param_change:.6f}")
            
            # Check if loss is increasing (potential overfitting or instability)
            if len(history['loss']) > 1 and avg_loss > history['loss'][-2] * 1.1:
                warning(f"⚠️ Loss increased by more than 10%: {history['loss'][-2]:.6f} -> {avg_loss:.6f}")
        
        info("Covariance head fine-tuning completed")
        return history
    
    def _get_motion_type_id(self, motion_type: str) -> int:
        """Get motion type ID from motion type string."""
        motion_type_map = {
            "car": 1,
            "dog": 2, 
            "drone": 3,
            "human": 4
        }
        return motion_type_map.get(motion_type, 3)  # Default to drone if not found
    
   
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model."""
        torch.save(self.model.state_dict(), save_path)
        info(f"Model saved to: {save_path}")
    
  


    def run_test_evaluation(self, test_data_path, motion_type="drone", output_dir="./test_results"):
        """
        Run the exact same evaluation as test.py by reusing all its functions.
        
        Args:
            test_data_path: Path to test data file or directory
            motion_type: Motion type ('car', 'dog', 'drone', 'human')
            output_dir: Output directory for results
            
        Returns:
            Evaluation results dictionary
        """
        # Create output directory
        self.out_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare test data path list in the same format as test.py
        if os.path.isfile(test_data_path):
            test_data_path_list = {motion_type: [test_data_path]}
        elif os.path.isdir(test_data_path):
            # Find all .npz files in the directory
            test_files = []
            for file in os.listdir(test_data_path):
                if file.endswith('.npz'):
                    test_files.append(os.path.join(test_data_path, file))
            test_data_path_list = {motion_type: test_files}
        else:
            raise ValueError(f"Test data path {test_data_path} does not exist")
        
        info(f"Running test evaluation on {len(test_data_path_list[motion_type])} files")
        
        # Create a test instance to reuse all its methods
        # Create a dummy args object for the tester
        class DummyArgs:
            def __init__(self):
                self.local_rank = 0
        
        args = DummyArgs()
        
        # Add missing fields to config
        if "test" not in self.config:
            self.config["test"] = {}
        self.config["test"]["out_dir"] = output_dir
        
        if "train" not in self.config:
            self.config["train"] = {}
        if "start_cov_epochs" not in self.config["train"]:
            self.config["train"]["start_cov_epochs"] = 0
            
        if "model" not in self.config:
            self.config["model"] = {}
        if "pred_velocity" not in self.config["model"]:
            self.config["model"]["pred_velocity"] = True
            
        if "model_param" not in self.config:
            self.config["model_param"] = {}
        if "window_time" not in self.config["model_param"]:
            self.config["model_param"]["window_time"] = 1.0
        
        test_instance = test.tester(args, self.config, self.model)
        
        # Run the exact same evaluation as test.py
        results = test_instance.test(test_data_path_list, epoch_num=0, resume_model=self.model, segment_length=5.0)
        
        success(f"Test evaluation completed. Results saved to: {output_dir}")
        return results


def main():
    """
    Example usage of the FineTuneInertialTracker with real data.
    """
    parser = argparse.ArgumentParser(description='Fine-tune Neural Inertial Tracking Model')
    parser.add_argument('--config', type=str, 
                       default="/neural_inertial_tracking/config/datasets/tartanimu/humanoid_train.yaml",
                       help='Path to configuration file')
    parser.add_argument('--model', type=str,
                       default="/exp_result/human_pretrain_addhuman_helmet/fine_tuned_model_epoch_5.pth",
                       help='Path to pre-trained model')
    parser.add_argument('--test_data', type=str,
                       default="/dataset/tartanimu_data/human/test/human_017.npz",
                       help='Path to test data file')
    parser.add_argument('--train_data', type=str,
                       default="/dataset/tartanimu_data/human/train",
                       help='Path to training data directory')
    parser.add_argument('--motion_type', type=str, default='drone',
                       choices=['car', 'dog', 'drone', 'human'],
                       help='Motion type for fine-tuning')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for fine-tuning (default: 1e-5 for conservative fine-tuning)')
    parser.add_argument('--debug_mode', action='store_true',
                       help='Debug mode: Freeze velocity head and train only covariance head (for sensor fusion)')

    parser.add_argument('--out_dir', type=str, default="/exp_result/human_pretrain_addhuman_helmet",
                       help='Path to output directory')
    
    args = parser.parse_args()
        # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialize tracker
    info("Initializing FineTuneInertialTracker...")
    tracker = FineTuneInertialTracker(args.config, args.model)
    
    # Example 1: Inference on real data
    success("=== Example 1: Inference on Real Data ===")
    
    results = tracker.inference_on_real_data(args.test_data, motion_type=args.motion_type)
    info(f"Prediction shape: {results['predictions'].shape}")
    info(f"Target shape: {results['targets'].shape}")
    
    # Example 2: Inference with covariance
    success("=== Example 2: Inference with Covariance ===")
    
    results_with_cov = tracker.inference_on_real_data(
        args.test_data, motion_type=args.motion_type, predict_covariance=True
    )
    info(f"Prediction shape: {results_with_cov['predictions'].shape}")
    info(f"Covariance shape: {results_with_cov['covariances'].shape}")
    
    # Example 3: Fine-tune covariance head
    success("=== Example 3: Fine-tune Covariance Head ===")
    
    # Find a training file
    train_files = []
    if os.path.isdir(args.train_data):
        for file in os.listdir(args.train_data):
            if file.endswith('.npz'):
                train_files.append(os.path.join(args.train_data, file))
    
    if train_files:
        train_file = train_files[0]  # Use first training file
        info(f"Using training file: {train_file}")
        
        history = tracker.fine_tune_covariance_head(
            train_file, motion_type=args.motion_type, 
            epochs=args.epochs, learning_rate=args.lr,
            debug_mode=args.debug_mode
        )
        if history['loss']:
            info(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
        else:
            info("No training performed (epochs=0)")
        
        # Save fine-tuned model with appropriate name
        if args.debug_mode:
            model_name = args.out_dir + "/fine_tuned_model_debug_epoch_" + str(args.epochs) + ".pth"
            info("💾 Saved debug mode model (velocity head frozen, covariance head trained)")
        else:
            model_name = args.out_dir + "/fine_tuned_model_epoch_" + str(args.epochs) + ".pth"
            info("💾 Saved normal mode model (covariance head trained)")
        
        tracker.save_model(model_name)
    else:
        warning("No training files found, skipping fine-tuning")
    
    # Example 4: Test Evaluation (exactly like test.py)
    success("=== Example 4: Test Evaluation (exactly like test.py) ===")
    

    
    test_results = tracker.run_test_evaluation(
        test_data_path=args.test_data,
        motion_type=args.motion_type,
        output_dir=args.out_dir
    )
    
    success(f"All results saved to: {args.out_dir}")
    
    success("=== All examples completed successfully! ===")


if __name__ == "__main__":
    main()
