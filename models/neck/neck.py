# models/neck/neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from quaternion.conv import QConv, QConv1d, QConv2d, QConv3d
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
import math 
import numpy as np
import torch
import torch.nn as nn


class QuaternionConcat(nn.Module):
    def __init__(self, dim=1, reduce=True, target_channels=None):
        """
        Initialize the QuaternionConcat module.

        Args:
            dim (int): The dimension along which to concatenate.
            reduce (bool): Whether to reduce the number of channels after concatenation.
            target_channels (int, optional): The desired number of channels after reduction.
        """
        super(QuaternionConcat, self).__init__()
        self.dim = dim
        self.reduce = reduce
        self.target_channels = target_channels

        if self.reduce:
            if self.target_channels is None:
                raise ValueError("target_channels must be specified when reduce=True")
            # 1x1 convolution to reduce channels, initialized later
            self.reduction_conv = None  # To be set dynamically in forward
            self.bn = None
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: list) -> torch.Tensor:
        """
        Forward pass for QuaternionConcat.

        Args:
            x (list): List of tensors to concatenate.

        Returns:
            torch.Tensor: Concatenated (and possibly reduced) tensor.
        """
        # Debug shapes
        # print("\nQuaternionConcat Input Shapes:")
        # for i, t in enumerate(x):
        #     print(f"Input tensor {i}: {t.shape}")

        # Ensure all tensors have batch dimension
        processed = []
        batch_size = None
        for t in x:
            if t.dim() == 3:
                t = t.unsqueeze(0)  # [1, C, H, W]
            elif t.dim() != 4:
                raise ValueError(f"Unexpected tensor shape: {t.shape}")
            processed.append(t)
            if batch_size is None:
                batch_size = t.shape[0]
            elif batch_size != t.shape[0]:
                raise ValueError("All tensors must have the same batch size")

        # Ensure spatial sizes match by cropping to the smallest height and width
        h_min = min(t.shape[2] for t in processed)
        w_min = min(t.shape[3] for t in processed)
        # Crop tensors to min height and width
        processed = [t[:, :, :h_min, :w_min] for t in processed]
        # Debug after cropping
        # for i, t in enumerate(processed):
        #     print(f"Input tensor {i} after cropping: {t.shape}")

        # Concatenate along the specified dimension
        out = torch.cat(processed, dim=self.dim)

        if self.reduce:
            if self.reduction_conv is None:
                in_channels = out.shape[1]
                self.reduction_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.target_channels,
                    kernel_size=1,
                    bias=False
                ).to(out.device)
                nn.init.kaiming_uniform_(self.reduction_conv.weight, a=math.sqrt(5))
                self.bn = nn.BatchNorm2d(self.target_channels).to(out.device)
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
            QConv2d(c, out_channels, kernel_size=1, stride=1, padding=0, bias=False) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            QConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) for _ in in_channels
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
            QConv2d(c, out_channels, kernel_size=3, stride=2, padding=1, bias=False) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            QConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) for _ in in_channels
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
