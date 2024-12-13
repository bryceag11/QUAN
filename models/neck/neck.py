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
        
        if self.reduce and self.target_channels is None:
            raise ValueError("`target_channels` must be specified when `reduce=True`.")
        
        self.reduction_conv = None
        self.bn = None
        self.relu = QReLU()

    def forward(self, x: list) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = [x]
            

        # Validate inputs
        for i, t in enumerate(x):
            if t.dim() != 5:
                raise ValueError(f"Tensor {i} has shape {t.shape}. Expected 5 dimensions [B, C, 4, H, W].")
            if t.shape[2] != 4:
                raise ValueError(f"Tensor {i} has quaternion dimension {t.shape[2]}, expected 4.")

        # Get largest spatial dimensions
        max_h = max(t.shape[-2] for t in x)
        max_w = max(t.shape[-1] for t in x)
        
        # Upsample smaller tensors to match largest spatial dimensions
        processed = []
        for t in x:
            if t.shape[-2] != max_h or t.shape[-1] != max_w:
                # Preserve the quaternion dimension during interpolation
                B, C, Q, H, W = t.shape
                t_reshaped = t.permute(0, 2, 1, 3, 4).reshape(B*Q, C, H, W)
                t_upsampled = F.interpolate(t_reshaped, size=(max_h, max_w), mode='nearest')
                t = t_upsampled.reshape(B, Q, C, max_h, max_w).permute(0, 2, 1, 3, 4)
            processed.append(t)

        # Concatenate along channel dimension
        out = torch.cat(processed, dim=1)  # [B, C*len(x), 4, H, W]

        if self.reduce:
            if self.reduction_conv is None:
                # For QConv2D, input_channels should be total channels divided by 4
                in_channels = out.shape[1] * 4
                
                self.reduction_conv = QConv2D(
                    in_channels=in_channels,
                    out_channels=self.target_channels,
                    kernel_size=1,
                    bias=False
                ).to(out.device)
                
                # BN works on channels divided by 4 due to quaternion structure
                self.bn = IQBN(self.target_channels // 4).to(out.device)
            
            # Apply reduction
            out = self.reduction_conv(out)
            out = self.bn(out)
            out = self.relu(out)

        return out

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
