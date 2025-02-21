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
from quaternion.conv import QConv2D, QDense
from quaternion.qactivation import QPReLU, QPReLU, QREReLU
import tqdm
import signal
import sys
import gc
from typing import OrderedDict
import math 

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
NUM_AUGMENTATIONS=3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'saved_models_final'
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
        mask = torch.rand(B, C, H, W, device=x.device, dtype=x.dtype) > self.p
        
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
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU()
        self.conv1 = QConv2D(in_planes, out_planes, kernel_size=3, stride=1,
                           padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU
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

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU()
        self.conv1 = QConv2D(in_planes, out_planes, kernel_size=1, stride=1,
                           padding=0, bias=False)
        self.droprate = dropRate
        self.pool = QuaternionAvgPool(kernel_size=2, stride=2)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = QuaternionDropout(p=self.droprate)(out)
        return self.pool(out)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class QuaternionDenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(QuaternionDenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        
        # First conv before any dense block - special handling for RGB input
        self.conv1 = QConv2D(3, in_planes, kernel_size=3, stride=1,
                           padding=1, bias=False, mapping_type='poincare')
        
        # Dense blocks
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, 
                                              int(math.floor(in_planes * reduction)), 
                                              dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, 
                                              int(math.floor(in_planes * reduction)), 
                                              dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        
        # Global average pooling and classifier
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU()
        self.fc = QDense(in_planes, num_classes * 4)  # *4 for quaternion output
        self.in_planes = in_planes
        self.global_pool = QuaternionAvgPool()


    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # Use quaternion global pooling
        out = self.global_pool(out)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        
        # Extract real components for final output
        batch_size = out.size(0)
        out = out.view(batch_size, -1, 4)  # Reshape to separate quaternion components
        return out[:, :, 0]  # Return real component [batch_size, num_classes]

def create_quaternion_densenet(depth=40, num_classes=10, growth_rate=12, dropRate=0.0):
    """Helper function to create a Quaternion DenseNet with standard configuration"""
    return QuaternionDenseNet(
        depth=depth,
        num_classes=num_classes,
        growth_rate=growth_rate,
        reduction=0.5,
        bottleneck=True,
        dropRate=dropRate
    )

class QuaternionBasicBlock(nn.Module):
    """Enhanced residual block for quaternion networks"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(QuaternionBasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = QConv2D(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1)
        self.bn1 = IQBN(out_channels)
        self.relu = nn.SiLU()

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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        # Second block
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.dropout2(out)
      
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
            QPReLU()
        )
        
        self.conv2_x = self._make_layer(64, 64, 3, 1, mapping_type, dropout_idx=0)
        self.conv3_x = self._make_layer(64, 128, 4, 2, mapping_type, dropout_idx=1)
        self.conv4_x = self._make_layer(128, 256, 6, 2, mapping_type, dropout_idx=2)
        self.conv5_x = self._make_layer(256, 256, 3, 2, mapping_type, dropout_idx=3)
        
        # Global Average Pooling: [3 × 128] output
        self.gap = QuaternionAvgPool()
        
        # Dropout before FC
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(256, 512, mapping_type=mapping_type),
            nn.ReLU(),
            nn.Dropout(p=self.current_rates[4]),  # Classifier dropout
            QDense(512, num_classes * 4, mapping_type=mapping_type)
        )
        # Final FC layer: 10 output classes
        # self.fc = nn.Linear(1024, num_classes)

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



class MultiAugmentDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies multiple augmentations to each image"""
    def __init__(self, dataset, augmentations_per_image=3, train=True):
        self.dataset = dataset
        self.augmentations_per_image = augmentations_per_image
        self.train = train
        
        # Strong augmentation for training
        self.strong_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Weak augmentation for one of the copies (maintains some original image properties)
        self.weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
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

def get_data_loaders(batch_size=256, augmentations_per_image=3, num_workers=8):
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

# def print_memory_stats():
#     print(f"\nMemory Stats:")
#     print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
#     process = psutil.Process()
#     print(f"CPU Memory: {process.memory_info().rss/1e9:.1f}GB")
#     print(f"CPU Usage: {psutil.cpu_percent()}%")

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
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
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
    
    # Initialize logging
    writer = SummaryWriter('runs/quaternion_densenet')
    metrics_logger = MetricsLogger(SAVE_DIR)
    
    # Get dataloaders with augmentation
    train_loader, test_loader = get_data_loaders(
        batch_size=BATCH_SIZE,
        augmentations_per_image=NUM_AUGMENTATIONS
    )
    model = QResNet34(num_classes=10, mapping_type='raw_normalized').to(DEVICE)
    # Print model parameter count
    num_params = count_parameters(model)
    print(f'\nTotal trainable parameters: {num_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.1,
                            momentum=0.9, 
                            weight_decay=1e-4,
                            nesterov=True)
    

    # class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    #     def __init__(self, optimizer, milestones, gamma=0.2, warmup_epochs=5, warmup_factor=0.1,
    #                 last_epoch=-1):
    #         self.milestones = milestones
    #         self.gamma = gamma
    #         self.warmup_epochs = warmup_epochs
    #         self.warmup_factor = warmup_factor
    #         super().__init__(optimizer, last_epoch)
    #     def get_lr(self):
    #         if self.last_epoch < self.warmup_epochs:
    #             # Linear warmup
    #             alpha = self.last_epoch / self.warmup_epochs
    #             warmup_factor = self.warmup_factor * (1 - alpha) + alpha
    #             return [base_lr * warmup_factor for base_lr in self.base_lrs]
    #         else:
    #             # Regular MultiStepLR behavior
    #             return [base_lr * self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])
    #                     for base_lr in self.base_lrs]
    #     def is_milestone(self, epoch):
    #         return epoch in self.milestones
        # def get_lr(self):
        #     if self.last_epoch < self.warmup_epochs:
        #         # Linear warmup
        #         alpha = self.last_epoch / self.warmup_epochs
        #         warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #         return [base_lr * warmup_factor for base_lr in self.base_lrs]
        #     else:
        #         # Regular MultiStepLR behavior
        #         return [base_lr * self.gamma ** len([m for m in self.milestones if m <= self.last_epoch])
        #                 for base_lr in self.base_lrs]
        # def is_milestone(self, epoch):
        #     return epoch in self.milestones

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=LEARNING_RATE,
    #     betas=(BETA_1, BETA_2),
    #     eps=EPSILON,
    #     weight_decay=5e-4
    # )
#     optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=0.005,  # Lower initial LR than SGD
#     betas=(0.9, 0.999),
#     eps=1e-7,
#     weight_decay=0.1
# )
    l1_reg = L1Regularization(L1_REG)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[75, 150, 225],
        gamma=0.1,
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=EPOCHS,
    #     eta_min=1e-6
    # )

#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.003,  # Match your initial LR
#     epochs=EPOCHS,
#     steps_per_epoch=len(train_loader),
#     pct_start=0.3,  # Warm up for 30% of training
#     div_factor=25,  # Initial LR will be max_lr/25
#     final_div_factor=1000,  # Final LR will be max_lr/1000
# )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=75,  # Initial restart interval
    #     T_mult=1,  # Multiply interval by 2 after each restart
    #     eta_min=1e-5
    # )
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=0.001,
#     epochs=EPOCHS,
#     steps_per_epoch=len(train_loader),
#     pct_start=0.3,
#     anneal_strategy='cos'
# )
    print('Starting training...')
    # print(f'Adam parameters: lr={LEARNING_RATE}, beta1={BETA_1}, beta2={BETA_2}')
    # print(f'Regularization: L1={L1_REG}, L2={L2_REG}')
    
    best_acc = 0
    pbar = tqdm.tqdm(total=EPOCHS, desc='Training Progress', position=0)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    

    for epoch in range(EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        
        # Validation
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
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