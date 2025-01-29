# models/neck/neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from quaternion.conv import QConv, QConv1D, QConv2D, QConv3D
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
import math 
import numpy as np
import torch
import torch.nn as nn



class QuaternionConcat(nn.Module):
    def __init__(self, dim=1, reduce=True, target_channels=None):
        super().__init__()
        self.dim = dim
        self.reduce = reduce
        self.target_channels = target_channels
        
        if reduce:
            assert target_channels is not None, "target_channels must be specified when reduce=True"
            assert target_channels % 4 == 0, "target_channels must be multiple of 4"
            # Create single quaternion convolution to reduce channels
            self.reduce_conv = QConv2D(target_channels * 4, target_channels, kernel_size=1)

    def forward(self, x: list) -> torch.Tensor:
        """
        Args:
            x: List of quaternion tensors each of shape [B, C, 4, H, W]
        Returns:
            torch.Tensor: Concatenated tensor [B, C', 4, H, W]
        """
        # Verify all inputs have quaternion structure
        assert all(tensor.size(2) == 4 for tensor in x), "All inputs must have quaternion dimension"
        
        # Concatenate along channel dimension while preserving quaternion structure
        concat = torch.cat(x, dim=1)  # [B, sum(C), 4, H, W]
        
        # Reduce channels if needed
        if self.reduce:
            concat = self.reduce_conv(concat)
            
        return concat

class QuaternionUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x):
        B, C, Q, H, W = x.shape
        
        # Reshape to handle quaternion components separately
        x = x.permute(0, 2, 1, 3, 4).reshape(B*Q, C, H, W)
        
        # Upsample
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        
        # Reshape back to quaternion format
        _, _, H_new, W_new = x.shape
        x = x.reshape(B, Q, C, H_new, W_new).permute(0, 2, 1, 3, 4)
        
        return x

class QuaternionFPN(nn.Module):
    """Feature Pyramid Network for Quaternion Neural Networks."""

    def __init__(self, in_channels, out_channels):
        super(QuaternionFPN, self).__init__()
        assert all(c % 4 == 0 for c in in_channels + [out_channels]), "Channels must be multiples of 4."
        
        self.lateral_convs = nn.ModuleList([
            QConv2D(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            QConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) for _ in in_channels
        ])
    
    def forward(self, inputs):
        """
        Forward pass through the FPN.
        
        Args:
            inputs (list): List of feature maps from the backbone.
        
        Returns:
            list: List of feature maps after FPN processing.
        """
        # Apply lateral convolutions
        lateral_feats = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        
        # Top-down pathway
        for i in range(len(lateral_feats) - 1, 0, -1):
            upsampled = F.interpolate(lateral_feats[i], scale_factor=2, mode='nearest')
            lateral_feats[i-1] += upsampled
        
        # Apply output convolutions
        out_feats = [output_conv(x) for output_conv, x in zip(self.output_convs, lateral_feats)]
        return out_feats

class QuaternionPAN(nn.Module):
    """Path Aggregation Network for Quaternion Neural Networks."""

    def __init__(self, in_channels, out_channels):
        super(QuaternionPAN, self).__init__()
        assert all(c % 4 == 0 for c in in_channels + [out_channels]), "Channels must be multiples of 4."
        
        self.down_convs = nn.ModuleList([
            QConv2D(c, out_channels, kernel_size=3, stride=2, padding=1, bias=False) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            QConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) for _ in in_channels
        ])
    
    def forward(self, inputs):
        """
        Forward pass through the PAN.
        
        Args:
            inputs (list): List of feature maps from FPN.
        
        Returns:
            list: List of feature maps after PAN processing.
        """
        # Bottom-up pathway
        for i in range(len(inputs) - 1):
            downsampled = self.down_convs[i](inputs[i])
            inputs[i+1] += downsampled
        
        # Apply output convolutions
        out_feats = [output_conv(x) for output_conv, x in zip(self.output_convs, inputs)]
        return out_feats
