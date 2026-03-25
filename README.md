# Offcial Implementation of TartanIMU

A clean, organized, and efficient implementation of neural inertial tracking using multi-head regression for IMU-based pose estimation.

## 🎯 Overview

This codebase provides a simplified and well-organized implementation of neural inertial tracking with the following features:

- **Multi-head regression** for different motion types (pen, watch, car, dog, drone, human)
- **ResNet-LSTM architecture** for robust feature extraction
- **Memory-optimized inference** with selective head computation
- **Centralized configuration management** with validation
- **Comprehensive error handling** and logging
- **Factory patterns** for clean object creation
- **Simplified training pipeline** with better organization

## 🏗️ Architecture

### **Core Components**
- **Model Factory**: Centralized model creation and management
- **Data Factory**: Efficient data loading and preprocessing
- **Config Manager**: Centralized configuration with validation
- **Error Handling**: Robust error recovery and validation
- **Logging System**: Comprehensive logging with colored output

### **Key Features**
- **Memory Optimization**: Only computes needed heads during inference
- **Multi-GPU Support**: Distributed training with NCCL backend
- **Mixed Precision**: Automatic mixed precision for faster training
- **Checkpoint Management**: Intelligent checkpoint saving and resuming
- **Validation**: Comprehensive input validation and error handling

## 🚀 Quick Start

### **1. Installation**

```bash
# Clone the repository
git clone <repository-url>
cd tartanimu

# Install dependencies
pip install -r requirement.txt

```

### **2. Data Preparation**

Organize your data in the following structure:
```
.
├── car
│   ├── 0
│   │   └── 0.npz
│   ├── 8
│   │   └── 8.npz
│   ├── test
│   │   └── 1087
│   │       └── 0.npz
│   ├── train
│   │   ├── 1091
│   │   │   ├── 1.npz
│   │   │   ├── 2.npz
│   │   │   ├── 3.npz
│   │   │   └── 4.npz
│   │   └── 112
│   │       └── 1.npz
│   └── val
│       └── 112
│           └── 1.npz
├── dog
│   ├── test
│   │   └── 2808
│   │       ├── 0.npz
│   │       ├── 1.npz
│   │       └── 2.npz
│   ├── train
│   │   └── 115
│   │       ├── 0.npz
│   │       └── 1.npz
│   └── val
│       └── 5326
│           └── 2.npz
├── drone
│   ├── test2
│   │   └── 5330
│   │       └── 0.npz
│   ├── train
│   │   └── 5340
│   │       └── 0.npz
│   └── val
│       └── 5390
│           ├── 0.npz
│           ├── 1.npz
│           └── 2.npz
└── human
    ├── test
    │   └── 5579
    │       ├── 1.npz
    │       ├── 2.npz
    │       ├── 3.npz
    │       ├── 4.npz
    │       ├── 5.npz
    │       └── 6.npz
    ├── train
    │   └── 5572
    │       └── 1.npz
    └── val
        ├── 5553
        │   └── 0.npz
        └── 5586
            ├── 0.npz
            ├── 1.npz
            └── 2.npz
```



### **2. Configuration Files**

The project uses YAML configuration files to manage training and inference settings. Here are the key configuration files:

#### **Dataset Configuration Files**
Located in `config/datasets/tartanimu/`:

**Training Configuration** (`humanoid.yaml`):

```yaml
# Training mode configuration
schemes:
  train: True   # Enable training
  test: False   # Disable testing during training
  online_adaption: False

train:
  out_dir: /path/to/training/output
  use_pretrain_model: False   # if you want to train from scratch, you need to set false. otherwise, set it true. 
```

**Inference Configuration** (`humanoid.yaml`):
```yaml
# Inference mode configuration
schemes:
  train: False   # Disable training
  test: True     # Enable testing/inference
  online_adaption: False

test:
  out_dir: /path/to/inference/output
```



### **3. Training and Inference Commands**

#### **Training/Testing Commands**
```bash
# Train with multi-head model
python main_net.py --config config/datasets/tartanimu/humanoid_train.yaml

# Test with multi-head model 
python main_net.py --config config/datasets/tartainimu/humanoid_test.yaml

```

####  **Finetune Existing Model**

```bash
python main_net.py --config config/datasets/tartanimu/humanoid_train.yaml --resume_from /mnt/shibo_intern_project/result/final_report/checkpoint_epoch_195.pt 

```

### **🚀 Minimal Example for Neural Inertial Tracking (Below might not work and I didn't test on my side)**

The `minimal_example.py` provides a clean, self-contained example for:
- **Model inference** on pre-trained neural inertial tracking models
- **Fine-tuning covariance heads** for sensor fusion applications  
- **Comprehensive evaluation** using the same metrics as `test.py`

#### **📋 Quick Start**

```bash
# Basic inference and evaluation
python minimal_example.py

# Fine-tune covariance head with custom parameters
python minimal_example.py --epochs 5 --lr 1e-5 --debug_mode

# Use custom data and model paths
python minimal_example.py \
    --config config/datasets/minimal_example/minimal_example_config.yaml \
    --model /path/to/your/pretrained_model.pt \
    --test_data /path/to/test/data.npz \
    --train_data /path/to/train/data/directory \
    --out_dir /path/to/output/results
```

##### **⚙️ Configuration Options**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `config/datasets/minimal_example/minimal_example_config.yaml` | Configuration file path |
| `--model` | Pre-trained model checkpoint | Path to pre-trained model |
| `--test_data` | Test data file/directory | Path to test data |
| `--train_data` | Training data directory | Path to training data |
| `--motion_type` | `drone` | Motion type: `car`, `dog`, `drone`, `human` |
| `--epochs` | `5` | Number of fine-tuning epochs |
| `--lr` | `1e-5` | Learning rate for fine-tuning |
| `--debug_mode` | `False` | Freeze velocity head, train only covariance |
| `--out_dir` | Output directory | Results output path |

##### **🎯 Use Cases**

###### **1. Model Inference Only**
```bash
# Run inference without fine-tuning
python minimal_example.py --epochs 0
```

###### **2. Covariance Head Fine-tuning**
```bash
# Fine-tune covariance head for sensor fusion
python minimal_example.py --epochs 10 --lr 1e-5 --debug_mode
```

###### **3. Overfitting Experiment**
```bash
# Test fine-tuning mechanism with same train/test data
python minimal_example.py \
    --test_data /path/to/data.npz \
    --train_data /path/to/data.npz \
    --epochs 3 --debug_mode
```

##### **📊 Output Structure**

```
output_directory/
├── fine_tuned_model_epoch_X.pth     # Fine-tuned model checkpoint
├── overall_summary.csv              # Overall evaluation metrics
├── trajectory_metrics.csv           # Per-trajectory results
├── segment_metrics.csv              # Segment-level analysis
├── statistical_summary.csv          # Statistical analysis
├── performance_comparison.csv       # Before/after comparison
└── plots/                          # Visualization plots
    ├── trajectory_plots/           # Trajectory visualizations
    ├── error_analysis/             # Error distribution plots
    └── covariance_analysis/        # Uncertainty analysis
```

##### **🔧 Advanced Configuration**

The minimal example uses a comprehensive YAML configuration file (`minimal_example_config.yaml`) that includes:

- **Model Architecture**: ResNet-LSTM parameters, output heads
- **Training Settings**: Batch sizes, learning rates, epochs
- **Fine-tuning**: Covariance head training, gradient clipping
- **Evaluation**: Metrics computation, visualization options
- **Performance**: Data loading, memory optimization

##### **💡 Tips for Best Results**

1. **Start with inference only** (`--epochs 0`) to verify model loading
2. **Use debug mode** (`--debug_mode`) for covariance-only fine-tuning
3. **Monitor gradient norms** in logs to detect training instability
4. **Check evaluation plots** for detailed performance analysis
5. **Use conservative learning rates** (1e-5 to 1e-6) for fine-tuning

##### **🔄 Integration with Sensor Fusion**

The fine-tuned covariance predictions can be integrated into differential factor graph frameworks:

```python
# Example integration workflow
1. Load fine-tuned model
2. Run inference to get velocity + covariance predictions
3. Feed covariance to sensor fusion (factor graph)
4. Use sensor fusion poses as improved ground truth
5. Iteratively fine-tune covariance head
```


## 📄 License

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 🙏 Acknowledgments

For technique details, please refer to TartanIMU[https://superodometry.com/tartanimu] and  research in IMU-based pose estimation and multi-head regression. 
