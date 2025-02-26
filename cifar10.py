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
from quaternion.qbatch_norm import IQBN, IQBN
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quaternion.conv import QConv2D, QDense, QConv
from quaternion.qactivation import QPReLU, QPReLU, QREReLU, QSiLU
import tqdm
import signal
import sys
import gc
from typing import OrderedDict
import math 
import random 

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
EPOCHS = 300
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7
L1_REG = 1e-5
L2_REG = 1e-4
DATA_AUGMENTATION = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'saved_models_feb'
MODEL_NAME = 'Q34_adamw.pth'

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
    
    def save(self, filename='Q34_admw_metrics.json'):
        """Save metrics to JSON file"""
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.metrics, f)
    
    def load(self, filename='Q34_admw_metrics.json'):
        """Load metrics from JSON file"""
        with open(self.save_dir / filename, 'r') as f:
            self.metrics = json.load(f)
    
    def plot(self, save_path='Q34_admw_plots.png'):
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
        mask = torch.rand(B, C, H, W, device=x.device) > self.p
       
        # Expand mask to all quaternion components (shape: B, C, 4, H, W)
        mask = mask.unsqueeze(2).expand(B, C, Q, H, W)
       
        # Apply mask and scale the output
        return x * mask / (1 - self.p)

class QuaternionAvgPool(nn.Module):
    """Quaternion-aware average pooling"""
    def __init__(self, kernel_size=None, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        if self.kernel_size is None:
            # Global average pooling
            pooled = F.adaptive_avg_pool2d(x_reshaped, (1, 1))
        else:
            # Strided pooling
            pooled = F.avg_pool2d(x_reshaped, 
                                kernel_size=self.kernel_size,
                                stride=self.stride)
        
        # Get output dimensions
        H_out, W_out = pooled.shape[-2:]
        
        # Reshape back to quaternion format (B, C, 4, H_out, W_out)
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

class QuaternionMaxPool(nn.Module):
    """Quaternion-aware max pooling"""
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
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


class BasicBlock(nn.Module):
    """Standard ResNet basic block"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity mapping or projection)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        
        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

class ResNet34(nn.Module):
    """
    Standard ResNet34 implementation following the original paper structure
    """
    def __init__(self, num_classes=10, small_input=True):
        super().__init__()
        
        # Initial layers - adapted for CIFAR-10 (small_input=True) or ImageNet (small_input=False)
        if small_input:  # For CIFAR-10
            # Single 3x3 conv for CIFAR sized images (32x32)
            self.initial_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:  # For ImageNet
            # Standard 7x7 conv followed by max pooling for ImageNet sized images
            self.initial_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        # ResNet blocks (layers)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final FC layer
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a ResNet layer composed of multiple BasicBlocks"""
        layers = []
        
        # First block with possible downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights (Kaiming initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolutional layer
        x = self.initial_layer(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification layer
        x = self.fc(x)
        
        return x
def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = IQBN(in_planes)
        self.relu = QSiLU()
        self.conv1 = QConv2D(in_planes, inter_planes, kernel_size=1, stride=1,
                           padding=0, bias=False)
        self.bn2 = IQBN(inter_planes)
        self.conv2 = QConv2D(inter_planes, out_planes, kernel_size=3, stride=1,
                           padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = QuaternionDropout(p=self.droprate)(out)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = QuaternionDropout(p=self.droprate)(out)
        return torch.cat([x, out], 1)


class QuaternionBasicBlock(nn.Module):
    """Enhanced residual block for quaternion networks"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(QuaternionBasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = QConv2D(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1)
        self.bn1 = IQBN(in_channels)
        self.relu = QSiLU()

        self.dropout1 = QuaternionDropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Second convolution block
        self.conv2 = QConv2D(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = IQBN(out_channels)

        self.dropout2 = QuaternionDropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Add batch normalization after shortcut for better regularization
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = QConv2D(in_channels, out_channels, kernel_size=1,
                                stride=stride)
    def forward(self, x):
        identity = self.shortcut(x)
        

        # First block
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        # out = self.bn1(out)
        # Second block
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)      
        out += identity
        return out

class QResNet34(nn.Module):
    """
    Quaternion ResNet34 implementation exactly matching the paper's architecture
    """
    def __init__(self, num_classes=10, mapping_type='raw_normalized'):
        super().__init__()
        
        self.dropout_rates = {
            'initial': [.1, .15, .2, .2, .25],  # [block2, block3, block4, block5, classifier]
            'increment': 0.05  # Amount to increase after each LR drop
        }
        self.current_rates = self.dropout_rates['initial'].copy()
        
        # Conv1: [3 × 32 × 32 × 16] output
        self.conv1 = nn.Sequential(
            QConv2D(3, 64, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(64),
            QSiLU()
        )
        
        self.conv2_x = self._make_layer(64, 64, 3, 1, mapping_type, dropout_idx=0)
        self.conv3_x = self._make_layer(64, 128, 4, 2, mapping_type, dropout_idx=1)
        self.conv4_x = self._make_layer(128, 256, 6, 2, mapping_type, dropout_idx=2)
        self.conv5_x = self._make_layer(256, 256, 3, 2, mapping_type, dropout_idx=3)
        
        # Global Average Pooling: [3 × 128] output
        self.gap = QuaternionAvgPool()
        
        if num_classes == 100:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                QDense(256, 512, mapping_type=mapping_type),
                nn.SiLU(),
                nn.Dropout(p=0.4),  # Increase dropout for regularization
                QDense(512, 1024, mapping_type=mapping_type),
                nn.SiLU(),
                nn.Dropout(p=0.4),
                QDense(1024, num_classes * 4, mapping_type=mapping_type)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                QDense(256, 512),             # First hidden layer
                nn.BatchNorm1d(512),
                nn.SiLU(),                    # Activation function
                nn.Dropout(p=0.3),              # Regularization
                QDense(512, 512),             # Second hidden layer
                nn.SiLU(),                    # Activation function
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),              # Regularization
                QDense(512, num_classes * 4)  # Final classification layer
            )


    def update_dropout_rates(self):
        """Increase dropout rates by the increment amount"""
        for i in range(len(self.current_rates)):
            self.current_rates[i] = min(0.5, self.current_rates[i] + self.dropout_rates['increment'])
            
        # Update dropout in all blocks
        self._update_block_dropout(self.conv2_x, 0)
        self._update_block_dropout(self.conv3_x, 1)
        self._update_block_dropout(self.conv4_x, 2)
        self._update_block_dropout(self.conv5_x, 3)
        
        # Update classifier dropout
        if isinstance(self.classifier[3], nn.Dropout):
            self.classifier[3].p = self.current_rates[4]

    def _update_block_dropout(self, block, rate_idx):
        """Update dropout rates in a block"""
        for layer in block:
            if isinstance(layer, QuaternionBasicBlock):
                layer.dropout1.p = self.current_rates[rate_idx]
                layer.dropout2.p = self.current_rates[rate_idx]


    def _make_layer(self, in_channels, out_channels, num_blocks, stride, mapping_type, dropout_idx):
        """Create a layer of residual blocks with dynamic dropout rates"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dropout_rate=self.current_rates[dropout_idx]  # Use current_rates with index
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                dropout_rate=self.current_rates[dropout_idx]  # Same dropout rate for all blocks in layer
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

        
        # Dropout
        x = self.classifier(x)

        batch_size = x.size(0)
        x = x.view(batch_size, NUM_CLASSES, 4)  # Reshape to separate quaternion components
        real_components = x[:, :, 0]  # Take only real part [batch_size, NUM_CLASSES]
        
        return real_components

class QuaternionBasicBlock_NoDrop(nn.Module):
    """Quaternion residual block without dropout"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(QuaternionBasicBlock_NoDrop, self).__init__()
        
        # First convolution block
        self.conv1 = QConv2D(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1)
        self.bn1 = IQBN(out_channels)
        self.relu = QSiLU()

        # Second convolution block
        self.conv2 = QConv2D(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = IQBN(out_channels)

        # Add batch normalization after shortcut for better regularization
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = QConv2D(in_channels, out_channels, kernel_size=1,
                                stride=stride)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second block
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        
        out += identity
        return out

# class QResNet34(nn.Module):
#     def __init__(self, num_classes=10, mapping_type='raw_normalized'):
#         super().__init__()
        
#         # Keep the existing convolutional backbone without dropout
#         self.conv1 = nn.Sequential(
#             QConv2D(3, 64, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
#             IQBN(64),
#             QSiLU()
#         )
        
#         # Remove dropout from these blocks
#         self.conv2_x = self._make_layer_without_dropout(64, 64, 3, 1, mapping_type)
#         self.conv3_x = self._make_layer_without_dropout(64, 128, 4, 2, mapping_type)
#         self.conv4_x = self._make_layer_without_dropout(128, 256, 6, 2, mapping_type)
#         self.conv5_x = self._make_layer_without_dropout(256, 256, 3, 2, mapping_type)
        
#         # Global Average Pooling
#         self.gap = QuaternionAvgPool()
        
#         # Enhanced classifier head for CIFAR-10
#         if num_classes == 10:
#             self.classifier = nn.Sequential(
#                 nn.Flatten(),
#                 # First dense block with BN
#                 QDense(256, 512, mapping_type=mapping_type),
#                 nn.BatchNorm1d(512*4),  # Apply BN after quaternion dense
#                 nn.SiLU(),
#                 nn.Dropout(0.2),  # Traditional dropout with lower rate
                
#                 # Second dense block
#                 QDense(512, 512, mapping_type=mapping_type),
#                 nn.BatchNorm1d(512*4),
#                 nn.SiLU(),
#                 nn.Dropout(0.2),
                
#                 # Output layer
#                 QDense(512, num_classes * 4, mapping_type=mapping_type)
#             )
#         else:
#             # Keep the existing CIFAR-100 classifier with dropout
#             self.classifier = nn.Sequential(
#                 nn.Flatten(),
#                 QDense(256, 512, mapping_type=mapping_type),
#                 nn.SiLU(),
#                 nn.Dropout(p=0.4),  # Increase dropout for regularization
#                 QDense(512, 1024, mapping_type=mapping_type),
#                 nn.SiLU(),
#                 nn.Dropout(p=0.4),
#                 QDense(1024, num_classes * 4, mapping_type=mapping_type)
#             )
            
    
#     def _make_layer_without_dropout(self, in_channels, out_channels, num_blocks, stride, mapping_type):
#         """Create a layer of residual blocks without dropout"""
#         layers = []
        
#         # First block handles stride and channel changes
#         layers.append(QuaternionBasicBlock_NoDrop(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             stride=stride
#         ))
        
#         # Remaining blocks
#         for _ in range(1, num_blocks):
#             layers.append(QuaternionBasicBlock_NoDrop(
#                 in_channels=out_channels,
#                 out_channels=out_channels,
#                 stride=1
#             ))
        
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         # Initial conv
#         x = self.conv1(x)
#         # Residual blocks
#         x = self.conv2_x(x)  # [3 × 32 × 32 × 16]
#         x = self.conv3_x(x)  # [3 × 16 × 16 × 32]
#         x = self.conv4_x(x)  # [3 × 8 × 8 × 64]
#         x = self.conv5_x(x)  # [3 × 4 × 4 × 128]
        
#         # Global average pooling
#         x = self.gap(x)  # [3 × 128]

        
#         # Dropout
#         x = self.classifier(x)

#         batch_size = x.size(0)
#         x = x.view(batch_size, NUM_CLASSES, 4)  # Reshape to separate quaternion components
#         real_components = x[:, :, 0]  # Take only real part [batch_size, NUM_CLASSES]
        
#         return real_components

class Cutout:
    """Randomly mask out a square patch from an image.
    
    Args:
        n_holes (int): Number of square patches to cut out.
        length (int): Length of the square side.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of shape (C, H, W).
        
        Returns:
            Tensor: Image with n_holes of specified length cut out.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), device=img.device)

        for n in range(self.n_holes):
            y = torch.randint(0, h, (1,))
            x = torch.randint(0, w, (1,))

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img

def visualize_feature_maps_to_disk(model, val_batch, epoch, output_dir):
    """
    Visualizes and saves feature maps to a directory
    
    Args:
        model: The model to visualize
        val_batch: A batch of validation data
        epoch: Current epoch number
        output_dir: Directory to save visualizations
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Only visualize every 10 epochs to avoid too many files
    if epoch % 10 != 0:
        return
    
    # Create output directory if it doesn't exist
    feature_maps_dir = os.path.join(output_dir, f'feature_maps/epoch_{epoch}')
    os.makedirs(feature_maps_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get first 8 images from batch
        images = val_batch[0][:8].to(DEVICE)
        
        # Forward pass through model layers
        # For QResNet34:
        x = model.conv1(images)
        
        # Save conv1 feature maps
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 10))
            # Get real part of quaternion features (first component)
            features = x[i, :, 0, :, :].cpu().mean(dim=0).numpy()
            
            # Normalize for better visualization
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
            
            ax.imshow(features, cmap='viridis')
            ax.set_title(f'Sample {i} - Conv1 Features')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_conv1.png'), dpi=150)
            plt.close(fig)
        
        # After conv2_x
        x = model.conv2_x(x)
        
        # Save conv2 feature maps
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 10))
            features = x[i, :, 0, :, :].cpu().mean(dim=0).numpy()
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
            
            ax.imshow(features, cmap='viridis')
            ax.set_title(f'Sample {i} - Conv2 Features')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_conv2.png'), dpi=150)
            plt.close(fig)
        
        # After conv3_x
        x = model.conv3_x(x)
        
        # Save conv3 feature maps
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 10))
            features = x[i, :, 0, :, :].cpu().mean(dim=0).numpy()
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
            
            ax.imshow(features, cmap='viridis')
            ax.set_title(f'Sample {i} - Conv3 Features')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_conv3.png'), dpi=150)
            plt.close(fig)
            
        # After conv4_x
        x = model.conv4_x(x)
        
        # Save conv4 feature maps
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 10))
            features = x[i, :, 0, :, :].cpu().mean(dim=0).numpy()
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
            
            ax.imshow(features, cmap='viridis')
            ax.set_title(f'Sample {i} - Conv4 Features')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_conv4.png'), dpi=150)
            plt.close(fig)
            
        # After conv5_x
        x = model.conv5_x(x)
        
        # Save conv5 feature maps
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(10, 10))
            features = x[i, :, 0, :, :].cpu().mean(dim=0).numpy()
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
            
            ax.imshow(features, cmap='viridis')
            ax.set_title(f'Sample {i} - Conv5 Features')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_conv5.png'), dpi=150)
            plt.close(fig)
            
        # Also save input images for reference
        for i in range(min(8, images.shape[0])):
            fig, ax = plt.subplots(figsize=(5, 5))
            # Denormalize the image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img * np.array([0.2675, 0.2565, 0.2761]) + np.array([0.5071, 0.4867, 0.4408]), 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Sample {i} - Input Image')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(feature_maps_dir, f'sample_{i}_input.png'), dpi=150)
            plt.close(fig)
            
    print(f"Feature maps for epoch {epoch} saved to {feature_maps_dir}")

class MultiAugmentDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies multiple augmentations to each image"""
    def __init__(self, dataset, augmentations_per_image=3, train=True):
        self.dataset = dataset
        self.augmentations_per_image = augmentations_per_image
        self.train = train
        
        # Import AutoAugment for CIFAR10
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        
        # Strong augmentation with AutoAugment and Cutout
        self.strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761)),
            Cutout(n_holes=1, length=16)
        ])
        
        # Weak augmentation with just basic transforms
        self.weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        ])
        
        # Test transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Pre-compute augmentation indices for efficiency
        if self.train:
            self.indices = []
            for idx in range(len(dataset)):
                self.indices.extend([idx] * augmentations_per_image)

    def __getitem__(self, index):
        if self.train:
            real_idx = self.indices[index]
            image, label = self.dataset[real_idx]
            
            # First augmentation is always weak, others are strong
            if index % self.augmentations_per_image == 0:
                transformed = self.weak_transform(image)
            else:
                transformed = self.strong_transform(image)
                
            return transformed, label
        else:
            image, label = self.dataset[index]
            return self.test_transform(image), label

    def __len__(self):
        if self.train:
            return len(self.dataset) * self.augmentations_per_image
        return len(self.dataset)

def get_data_loaders(batch_size=256, augmentations_per_image=1, num_workers=1):
    """Get train and test data loaders with multiple augmentations"""
    
    # Load base CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None)
    
    # Wrap datasets with multi-augmentation
    train_dataset = MultiAugmentDataset(
        trainset, 
        augmentations_per_image=augmentations_per_image,
        train=True
    )
    test_dataset = MultiAugmentDataset(
        testset,
        augmentations_per_image=1,  # No augmentation for test
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  
        persistent_workers=True,  
        prefetch_factor=3,
        drop_last=True
    ) 
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  
        persistent_workers=True, 
        prefetch_factor=3,
        drop_last=True 
    )
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """
    Train for one epoch with optimized GPU handling
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm.tqdm(train_loader, desc='Training', position=1, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(train_pbar):
        # Move data to GPU efficiently
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than standard zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Clean up GPU memory
        del outputs, loss
        
    train_pbar.close()
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate with optimized GPU handling
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            del outputs, loss
    
    return test_loss / len(test_loader), 100. * correct / total

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Constants for augmentation
    NUM_AUGMENTATIONS = 3  # Number of augmentations per image
    # RANDOM_SEED = 42
    # set_random_seeds(RANDOM_SEED)
    # Initialize logging
    writer = SummaryWriter('runs/quaternion_densenet')
    metrics_logger = MetricsLogger(SAVE_DIR)
    
    # Get dataloaders with augmentation
    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        augmentations_per_image=1,
        num_workers=1,
    )
    # model = ResNet34(num_classes=10)
    model = QResNet34(num_classes=10, mapping_type='poincare')
    # Print model parameter count
    num_params = count_parameters(model)
    print(f'\nTotal trainable parameters: {num_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.1,
                            momentum=0.9, 
                            weight_decay=5e-4,
                            nesterov=True)
    

    l1_reg = L1Regularization(L1_REG)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[150, 225],
        gamma=0.1,
    )


    print('Starting training...')
    # print(f'Adam parameters: lr={LEARNING_RATE}, beta1={BETA_1}, beta2={BETA_2}')
    # print(f'Regularization: L1={L1_REG}, L2={L2_REG}')
    
    best_acc = 0
    pbar = tqdm.tqdm(total=EPOCHS, desc='Training Progress', position=0)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    feature_maps_dir = os.path.join(SAVE_DIR, 'feature_maps')
    os.makedirs(feature_maps_dir, exist_ok=True)
    val_batch = next(iter(test_loader))

    for epoch in range(EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        
        # Validation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        visualize_feature_maps_to_disk(model, val_batch, epoch, SAVE_DIR)
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update progress bar
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
            'train_loss': train_loss,
            'test_loss': test_loss,
        })
        
        # TensorBoard logging
        writer.add_scalar('learning_rate', current_lr, epoch)
        writer.add_scalar('training/accuracy', train_acc, epoch)
        writer.add_scalar('test/accuracy', test_acc, epoch)
        writer.add_scalar('training/loss', train_loss, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)
        
        # Save metrics visualization
        if (epoch + 1) % 10 == 0:
            metrics_logger.save()
            metrics_logger.plot()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'\nSaving model (acc: {test_acc:.2f}%)')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': best_acc,
            }, os.path.join(SAVE_DIR, MODEL_NAME))
        
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
    
    pbar.close()
    metrics_logger.plot('final_metrics.png')
    writer.close()
    
    print(f'Best test accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()