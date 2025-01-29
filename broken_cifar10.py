#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from quaternion.qbatch_norm import QBN, IQBN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quaternion.conv import QConv2D, QDense
import tqdm
import signal
import sys
import gc
from typing import OrderedDict

def handle_keyboard_interrupt(signum, frame):
    """
    Custom handler for keyboard interrupt to ensure clean exit
    """
    print("\n\nTraining interrupted by user. Cleaning up...")
    
    # Attempt to close any open progress bars
    try:
        # If you're using nested progress bars, close them
        if 'pbar' in globals():
            pbar.close()
        if 'train_pbar' in globals():
            train_pbar.close()
        if 'test_pbar' in globals():
            test_pbar.close()
    except Exception as e:
        print(f"Error closing progress bars: {e}")
    
    # Close TensorBoard writer if it exists
    try:
        if 'writer' in globals():
            writer.close()
    except Exception as e:
        print(f"Error closing TensorBoard writer: {e}")
    
    # Optional: Save current model state
    try:
        if 'model' in globals() and 'optimizer' in globals():
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'interrupt_save': True
            }, 'interrupt_checkpoint.pth')
            print("Saved interrupt checkpoint.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    
    # Exit cleanly
    sys.exit(0)

# Register the keyboard interrupt handler
signal.signal(signal.SIGINT, handle_keyboard_interrupt)



# Add parameter counting function
def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




# Configuration
BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 150
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7
L1_REG = 1e-5
L2_REG = 1e-4
DATA_AUGMENTATION = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'saved_models'
MODEL_NAME = 'pytorch_cifar10_quaternion.pth'

class L1Regularization:
    """L1 regularization for network parameters"""
    def __init__(self, l1_lambda):
        self.l1_lambda = l1_lambda
        
    def __call__(self, model):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_reg = l1_reg + torch.sum(torch.abs(param))
        return self.l1_lambda * l1_reg



class QuaternionDenseLayer(nn.Module):
    """Single layer in quaternion DenseNet"""
    def __init__(self, in_channels, growth_rate, bn_size=4, mapping_type='luminance'):
        super().__init__()
        
        bottleneck_channels = ((bn_size * growth_rate) // 4) * 4
        
        self.bn1 = IQBN(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = QConv2D(in_channels, bottleneck_channels, 
                        kernel_size=1, bias=False, mapping_type=mapping_type)
        
        # Ensure growth rate is multiple of 4
        growth_rate = (growth_rate // 4) * 4
        
        self.bn2 = IQBN(bottleneck_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = QConv2D(bottleneck_channels, growth_rate,
                        kernel_size=3, padding=1, bias=False, mapping_type=mapping_type)
        
    def forward(self, x):
        # Bottleneck
        out = self.conv1(self.relu1(self.bn1(x)))
        # Main conv
        out = self.conv2(self.relu2(self.bn2(out)))
        # Dense connection
        return torch.cat([x, out], 1)

class QuaternionDenseBlock(nn.Module):
    """Dense block with multiple densely connected layers"""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, mapping_type='luminance'):
        super().__init__()
        self.layers = nn.ModuleList([
            QuaternionDenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                mapping_type
            ) for i in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class QuaternionTransition(nn.Module):
    """Transition layer between dense blocks with compression"""
    def __init__(self, in_channels, out_channels, mapping_type='luminance'):
        super().__init__()
        # Ensure output channels are multiple of 4
        out_channels = (out_channels // 4) * 4
        
        self.bn = IQBN(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = QConv2D(in_channels, out_channels,
                        kernel_size=1, bias=False, mapping_type=mapping_type)
        self.pool = QuaternionAvgPool()
        
    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class QDenseNetBC(nn.Module):
    """
    Quaternion DenseNet-BC implementation
    BC refers to bottleneck layers and compression factor
    
    Args:
        growth_rate (int): Growth rate (k) - number of feature maps produced by each layer
        block_config (tuple): Number of layers in each dense block
        compression (float): Compression factor (0-1) for transition layers
        num_init_features (int): Number of initial convolution filters
        bn_size (int): Expansion factor for bottleneck layer
        num_classes (int): Number of output classes
        mapping_type (str): Type of quaternion mapping to use
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 compression=0.5, num_init_features=64, bn_size=4,
                 num_classes=10, mapping_type='luminance'):
        super().__init__()
        
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', QConv2D(3, num_init_features, kernel_size=3, 
                           padding=1, bias=False, mapping_type=mapping_type)),
            ('norm0', IQBN(num_init_features)),
            ('relu0', nn.ReLU(inplace=True))
        ]))
        
        # Dense blocks
        num_features = num_init_features
        # Ensure initial features are multiple of 4
        num_init_features = (num_init_features // 4) * 4
        # Ensure growth rate is multiple of 4
        growth_rate = (growth_rate // 4) * 4
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = QuaternionDenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                mapping_type=mapping_type
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Add transition layer except after last block
            if i != len(block_config) - 1:
                num_output_features = int(num_features * compression)
                trans = QuaternionTransition(
                    in_channels=num_features,
                    out_channels=num_output_features,
                    mapping_type=mapping_type
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_output_features
        
        # Final batch norm
        self.features.add_module('norm_final', IQBN(num_features))
        self.features.add_module('relu_final', nn.ReLU(inplace=True))
        
        # Global Average Pooling and classifier
        self.gap = QuaternionAvgPool()
        self.classifier = nn.Linear(num_features, num_classes)
        

    
    def forward(self, x):
        features = self.features(x)
        out = self.gap(features)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

def create_qdensenet(growth_rate=24, num_classes=10):
    """Factory function to create quaternion DenseNet-BC with typical configs"""
    return QDenseNetBC(
        growth_rate=growth_rate,
        block_config=(6, 12, 24, 16),  # Standard BC config
        compression=0.5,
        num_init_features=64,
        bn_size=4,
        num_classes=num_classes
    )

class MetricsLogger:
    """Logger for training and evaluation metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': [],
            'train_reg_loss': [],
            'test_reg_loss': []
        }
    
    def update(self, epoch_metrics):
        """Update metrics with new values"""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save(self, filename='metrics.json'):
        """Save metrics to JSON file"""
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.metrics, f)
    
    def load(self, filename='metrics.json'):
        """Load metrics from JSON file"""
        with open(self.save_dir / filename, 'r') as f:
            self.metrics = json.load(f)
    
    def plot(self, save_path='training_plots.png'):
        """Create and save visualization plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        epochs = range(1, len(self.metrics['train_acc']) + 1)
        
        # Accuracy plot
        ax1.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, self.metrics['test_acc'], 'r-', label='Test Accuracy')
        ax1.set_title('Training and Test Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, self.metrics['test_loss'], 'r-', label='Test Loss')
        ax2.plot(epochs, self.metrics['train_reg_loss'], 'g--', label='Train Reg Loss')
        ax2.plot(epochs, self.metrics['test_reg_loss'], 'y--', label='Test Reg Loss')
        ax2.set_title('Training and Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_path)
        plt.close()

class QuaternionDropout(nn.Module):
    """
    Applies the same dropout mask across all four components of a quaternion tensor.
    
    Args:
        p (float): Probability of an element to be zeroed. Default: 0.5.
    """
    def __init__(self, p=0.5):
        super(QuaternionDropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x  # No dropout during evaluation or if p=0
        
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components."
        
        # Generate dropout mask for one quaternion component (shape: B, C, H, W)
        mask = torch.rand(B, C, H, W, device=x.device, dtype=x.dtype) > self.p
        
        # Expand mask to all quaternion components (shape: B, C, 4, H, W)
        mask = mask.unsqueeze(2).expand(B, C, Q, H, W)
        
        # Apply mask and scale the output
        return x * mask / (1 - self.p)


class QuaternionResNet(nn.Module):
    """Deeper ResNet-style architecture with quaternion convolutions based on the paper's implementation"""
    def __init__(self, num_classes=10, mapping_type='luminance'):
        super(QuaternionResNet, self).__init__()
        
        # Initial convolution layer - matches Conv1 from Table II
        self.conv1 = nn.Sequential(
            QConv2D(3, 16, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(16),
            nn.ReLU(inplace=True)
        )
        
        # Conv2_x block - 3 residual blocks (16 channels)
        self.conv2_x = self._make_layer(16, 16, blocks=3, stride=1, mapping_type=mapping_type)
        
        # Conv3_x block - 4 residual blocks (32 channels)
        self.conv3_x = self._make_layer(16, 32, blocks=4, stride=2, mapping_type=mapping_type)
        
        # Conv4_x block - 6 residual blocks (64 channels)
        self.conv4_x = self._make_layer(32, 64, blocks=6, stride=2, mapping_type=mapping_type)
        
        # Conv5_x block - 3 residual blocks (128 channels)
        self.conv5_x = self._make_layer(64, 128, blocks=3, stride=2, mapping_type=mapping_type)
        
        # Quaternion-aware global average pooling
        self.avg_pool = QuaternionAvgPool()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Final fully connected layer
        self.fc = QDense(128, num_classes * 4, mapping_type=mapping_type)

    def _make_layer(self, in_channels, out_channels, blocks, stride, mapping_type):
        """Create a layer of residual blocks"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(in_channels, out_channels, stride, mapping_type))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(out_channels, out_channels, 1, mapping_type))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # Residual blocks
        x = self.conv2_x(x)  # 3 blocks
        x = self.conv3_x(x)  # 4 blocks
        x = self.conv4_x(x)  # 6 blocks
        x = self.conv5_x(x)  # 3 blocks
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Flatten
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, 4, 32, 1, 1]
        x = x.view(x.size(0), 4, -1)  # [B, 4, 32]
        x = x.permute(0, 2, 1).contiguous()  # [B, 32, 4]
        x = x.reshape(x.size(0), -1)  # [B, 128]
        
        # Two options for classification:
        # Option 1: Using QDense (maintain quaternion structure)
        x = self.fc(x)  # QDense output: [B, num_classes * 4]
        
        # Reshape to quaternion format and take real part
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)  # [B, num_classes, 4]
        return x[:, :, 0]  # Return real component [B, num_classes]
        

class QuaternionAvgPool(nn.Module):
    """Quaternion-aware average pooling"""
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        pooled = self.avg_pool(x_reshaped)
        
        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

class QuaternionMaxPool(nn.Module):
    """Quaternion-aware max pooling"""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        pooled = self.pool(x_reshaped)
        
        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

class QuaternionBasicBlock(nn.Module):
    """Enhanced residual block for quaternion networks"""
    def __init__(self, in_channels, out_channels, stride=1, mapping_type='luminance'):
        super(QuaternionBasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = QConv2D(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, mapping_type=mapping_type)
        self.bn1 = IQBN(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution block
        self.conv2 = QConv2D(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, mapping_type=mapping_type)
        self.bn2 = IQBN(out_channels)
        
        # Add batch normalization after shortcut for better regularization
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                QConv2D(in_channels, out_channels, kernel_size=1,
                        stride=stride, mapping_type=mapping_type),
                IQBN(out_channels)
            )

    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out



class QResNet34(nn.Module):
    """
    Quaternion ResNet34 implementation exactly matching the paper's architecture
    """
    def __init__(self, num_classes=10, mapping_type='luminance'):
        super().__init__()
        
        # Conv1: [3 × 32 × 32 × 16] output
        self.conv1 = nn.Sequential(
            QConv2D(3, 16, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(16),
            nn.ReLU(inplace=True)
        )
        
        # Conv2_x: 3 blocks, [3 × 32 × 32 × 16] output
        self.conv2_x = self._make_layer(
            in_channels=16,
            out_channels=16,
            num_blocks=3,
            stride=1,
            mapping_type=mapping_type
        )
        
        # Conv3_x: 4 blocks, [3 × 16 × 16 × 32] output
        self.conv3_x = self._make_layer(
            in_channels=16,
            out_channels=32,
            num_blocks=4,
            stride=2,
            mapping_type=mapping_type
        )
        
        # Conv4_x: 6 blocks, [3 × 8 × 8 × 64] output
        self.conv4_x = self._make_layer(
            in_channels=32,
            out_channels=64,
            num_blocks=6,
            stride=2,
            mapping_type=mapping_type
        )
        
        # Conv5_x: 3 blocks, [3 × 4 × 4 × 128] output
        self.conv5_x = self._make_layer(
            in_channels=64,
            out_channels=128,
            num_blocks=3,
            stride=2,
            mapping_type=mapping_type
        )
        
        # Global Average Pooling: [3 × 128] output
        self.gap = QuaternionAvgPool()
        
        # Dropout before FC
        self.dropout = nn.Dropout(0.5)
        
        # Final FC layer: 10 output classes
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, mapping_type):
        """Create a layer of residual blocks"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            mapping_type=mapping_type
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                mapping_type=mapping_type
            ))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual blocks
        x = self.conv2_x(x)  # [3 × 32 × 32 × 16]
        x = self.conv3_x(x)  # [3 × 16 × 16 × 32]
        x = self.conv4_x(x)  # [3 × 8 × 8 × 64]
        x = self.conv5_x(x)  # [3 × 4 × 4 × 128]
        
        # Global average pooling
        x = self.gap(x)  # [3 × 128]
        x = x.reshape(x.size(0), -1)  # Flatten
        
        # Dropout
        x = self.dropout(x)
        
        # Classification
        x = self.fc(x)
        
        return x

class QuaternionCIFAR10(nn.Module):
    """
    Quaternion CNN model for CIFAR-10 classification.
    """
    def __init__(self, mapping_type='luminance'):
        super(QuaternionCIFAR10, self).__init__()
    
        # PREVIOUSLY WORKING CODE
        # First block
        # Initial convolution block
        self.initial_block = nn.Sequential(
            QConv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(32),
            nn.ReLU(),
        )

        # First dense block
        self.block1 = nn.Sequential(
            QConv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(32),
            nn.ReLU(),
            QConv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            # IQBN(32),
            nn.ReLU(),
        )

        # Second convolution block
        self.block2 = nn.Sequential(
            QConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(64),
            nn.ReLU(),
            QConv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, mapping_type=mapping_type),
            # IQBN(64),
            nn.ReLU(),
        )

        # Third convolution block
        self.block3 = nn.Sequential(
            QConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(128),
            nn.ReLU(),
            QConv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(128),
            nn.ReLU(),
        )

        # Fourth convolution block
        self.block4 = nn.Sequential(
            QConv2D(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(256),
            nn.ReLU(),
            QConv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(256),
            nn.ReLU(),
        )

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.dropout = QuaternionDropout(p=0.1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(256, 512, mapping_type=mapping_type),
            nn.ReLU(),
            nn.Dropout(0.3),
            QDense(512, NUM_CLASSES * 4, mapping_type=mapping_type)  # Output 4x classes for quaternion
        )
    
    def pool_spatial_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies spatial pooling independently for each quaternion component.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
            pool_layer (nn.MaxPool2d): Pooling layer to apply.

        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, 4, H_out, W_out).
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components."

        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)

        # Apply pooling
        pooled = self.pool(x_reshaped)

        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

    def avg_pool(self, x: torch.Tensor, num) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components."

        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)

        # Apply pooling
        pooled = F.adaptive_avg_pool2d(x_reshaped, (num, num))

        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # First block with dropout
        x = self.block1(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Second block
        x = self.block2(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Third block
        x = self.block3(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Fourth block
        x = self.block4(x)
        x = self.pool_spatial_only(x)  # Alternate between avg and spatial pooling
        x = self.dropout(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Extract only real components for final classification
        batch_size = x.size(0)
        x = x.view(batch_size, NUM_CLASSES, 4)  # Reshape to separate quaternion components
        real_components = x[:, :, 0]  # Take only real part [batch_size, NUM_CLASSES]
        
        return real_components

def worker_init_fn(worker_id):
    torch.cuda.empty_cache()
    # Optional: force garbage collection
    gc.collect()

def get_data_loaders():
    """
    Prepare CIFAR-10 data loaders with augmentation.
    """
    if DATA_AUGMENTATION:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12,
          pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True,
          persistent_workers=True, worker_init_fn=worker_init_fn)

    return trainloader, testloader

def evaluate(model, test_loader, criterion, l1_reg, epoch, writer):
    """Evaluate the model with proper memory handling."""
    model.eval()
    test_loss = 0
    test_reg_loss = 0
    correct = 0
    total = 0
    
    test_pbar = tqdm.tqdm(test_loader, desc='Testing', position=1, leave=False)
    
    with torch.no_grad():
        for inputs, targets in test_pbar:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            criterion_loss = criterion(outputs, targets)
            l1_loss = l1_reg(model)
            
            # Update metrics
            test_loss += criterion_loss.item()
            test_reg_loss += l1_loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            test_pbar.set_postfix({
                'Loss': f'{test_loss/total:.4f}',
                'Accuracy': f'{current_acc:.2f}%'
            })
            
            # Clear memory
            del outputs, criterion_loss, l1_loss, predicted
            
    test_pbar.close()
    
    # Calculate final metrics
    test_acc = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    avg_test_reg_loss = test_reg_loss / len(test_loader)
    
    return test_acc, avg_test_loss, avg_test_reg_loss

def train_epoch(model, train_loader, criterion, optimizer, l1_reg, epoch):
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm.tqdm(train_loader, desc='Training', position=1, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(train_pbar):
        # Move data to device
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        # Clear gradients and optimizer state
        for param in model.parameters():
            param.grad = None  # More efficient than zero_grad()
        
        # Instead of clearing gradients for each parameter
        optimizer.zero_grad(set_to_none=True)  # More efficient
        
        outputs = model(inputs)
        criterion_loss = criterion(outputs, targets)
        l1_loss = l1_reg(model)
        total_loss = criterion_loss + l1_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        running_loss += criterion_loss.item()
        running_reg_loss += l1_loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Explicitly clear memory
        del outputs, criterion_loss, l1_loss, total_loss
        if batch_idx % 10 == 0:  # Periodic cleanup
            torch.cuda.empty_cache()
            
        # Update progress bar
        current_acc = 100. * correct / total
        train_pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Reg Loss': f'{running_reg_loss/(batch_idx+1):.4f}',
            'Accuracy': f'{current_acc:.2f}%'
        })
    
    train_pbar.close()
    
    # Calculate final metrics
    train_acc = 100. * correct / total
    avg_train_loss = running_loss / len(train_loader)
    avg_train_reg_loss = running_reg_loss / len(train_loader)
    
    return train_acc, avg_train_loss, avg_train_reg_loss



def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    writer = SummaryWriter('runs/cifar10_quaternion_adam')
    metrics_logger = MetricsLogger(SAVE_DIR)
    
    train_loader, test_loader = get_data_loaders()
    
    model = QResNet34(num_classes=10, mapping_type='luminance').to(DEVICE)
    # Print model parameter count
    num_params = count_parameters(model)
    print(f'\nTotal trainable parameters: {num_params:,}')

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), 
    #                         lr=0.1,
    #                         momentum=0.9, 
    #                         weight_decay=5e-4,
    #                         nesterov=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA_1, BETA_2),
        eps=EPSILON,
        weight_decay=L2_REG
    )
    l1_reg = L1Regularization(L1_REG)
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    # optimizer,
    # milestones=[100, 150],  # As per DenseNet paper
    # gamma=0.1  # Reduces LR by 10x at these epochs
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )
    print('Starting training...')
    # print(f'Adam parameters: lr={LEARNING_RATE}, beta1={BETA_1}, beta2={BETA_2}')
    # print(f'Regularization: L1={L1_REG}, L2={L2_REG}')
    
    best_acc = 0
    pbar = tqdm.tqdm(total=EPOCHS, desc='Training Progress', position=0)
    
    for epoch in range(EPOCHS):
        # if epoch % 5 == 0:  # Flush writer periodically
        #     writer.flush()
        # Training
        torch.cuda.empty_cache()
        gc.collect()
        
        train_acc, avg_train_loss, avg_train_reg_loss = train_epoch(
            model, train_loader, criterion, optimizer, l1_reg, epoch)
            
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            
        test_acc, avg_test_loss, avg_test_reg_loss = evaluate(
            model, test_loader, criterion, l1_reg, epoch, writer)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        pbar.update(1)
        pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Test Acc': f'{test_acc:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
        
        # Log metrics
        metrics_logger.update({
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_reg_loss': avg_train_reg_loss,
            'test_reg_loss': avg_test_reg_loss
        })
        
        # Log to tensorboard
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('test accuracy', test_acc, epoch)
        writer.add_scalar('train loss', avg_train_loss, epoch)
        writer.add_scalar('test loss', avg_test_loss, epoch)
        # Create visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics_logger.save()
            metrics_logger.plot()      
        # Save best model
        # if test_acc > 85.00:
        if epoch % 10 == 0:
            if test_acc > best_acc:
                print(f'\nSaving model (acc: {test_acc:.2f}%)')
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': best_acc,
                }, os.path.join(SAVE_DIR, MODEL_NAME))
        
        # Force garbage collection
        torch.cuda.empty_cache()
        gc.collect()
    
    pbar.close()
    metrics_logger.plot('final_training_plots.png')

    writer.flush() 
    writer.close()
    
    print(f'Best test accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()