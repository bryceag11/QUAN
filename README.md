# QUAN: Quaternion Approximation Networks 
## ![Please visit our newest repo](https://github.com/bryceag11/QUAN_ultralytics)
A modular, configurable deep learning framework for training and evaluating quaternion neural networks on image classification and object detection datasets. This framework supports both standard convolutional neural networks and quaternion neural networks.

![figure2_poincare_visualization](https://github.com/user-attachments/assets/8b4b083b-d657-43cf-8cf5-a2d6cdb8af46)


## Project Structure

```
QuatNet_OBB/
├── config/
│   ├── default.yaml
│   ├── default_detection.yaml
│   └── experiments/
├── models/
│   ├── registry.py
│   ├── blocks/
│   │   ├── blocks.py
│   │   ├── neck.py
│   │   ├── head.py
│   │   └── qblocks.py
│   ├── architectures/
│   │   └── blocks.py
├── utils/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── checkpoint.py
├── experiments/
├── losses/
│   └── loss.py
└── train.py
```

## Installation

```bash
# Clone the repository
git clone https://github.com/bryceag11/QuatNet_OBB.git
cd cifar-training

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```

## Usage

### Basic Training

```bash
# Train with default configuration
python train.py

# Train with a specific experiment configuration
python train.py --config config/experiments/qresnet34_custom.yaml

# Resume training from the latest checkpoint
python train.py --config config/experiments/qresnet34_custom.yaml --resume
```

### Creating a Custom Configuration

1. Create a new YAML file in `config/experiments/`
2. Modify the parameters as needed
3. Run training with the new configuration

Example configuration:

```yaml
# config/experiments/my_experiment.yaml

experiment:
  name: "MyExperiment"
  description: "Custom experiment configuration"
  seed: 42

dataset:
  name: "cifar10"
  batch_size: 128
  augmentations_per_image: 2

model:
  type: "qresnet"
  name: "qresnet34"
  mapping_type: "poincare"
  
  # Custom block specification
  blocks:
    - type: "conv"
      out_channels: 64
      kernel_size: 3
    # ... more blocks ...
```

## Creating Custom Architectures

You can create custom architectures in two ways:

1. **Using Block Specification**: Define the network structure using blocks in the YAML configuration
2. **Creating a New Architecture Class**: Implement a new architecture in the `models/architectures/` directory

### Example: Custom Block Specification

```yaml
model:
  type: "qresnet"
  name: "custom_model"
  blocks:
    # Initial convolution
    - type: "conv"
      out_channels: 64
      kernel_size: 3
    
    # Residual blocks
    - type: "basic"
      out_channels: 128
      num_blocks: 3
      stride: 2
    
    # Global pooling
    - type: "global_pool"
    
    # Flatten
    - type: "flatten"
    
    # Final classifier
    - type: "fc"
      out_features: 10
```

## Components

## Quaternion Networks

This framework supports quaternion neural networks which represent each weight as a quaternion number with 4 components. 

To enable quaternion operations, set the appropriate component types in your configuration:

```yaml
components:
  conv_type: "QConv2D"
  norm_type: "IQBN"
  activation_type: "QSiLU"
  pooling_type: "QuaternionAvgPool"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
