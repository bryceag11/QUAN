# models/neck/neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1d, QConv2d, QConv3d
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN


class QuaternionConcat(nn.Module):
    """
    Concatenation layer for quaternion tensors.
    """
    def __init__(self, dim=1):
        super(QuaternionConcat, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Concatenates a list of tensors along the specified dimension.
        
        Args:
            x (list): List of tensors to concatenate. Each tensor should have channel counts as multiples of 4.
        
        Returns:
            torch.Tensor: Concatenated tensor.
        """
        # Verify all tensors have channel multiples of 4
        for tensor in x:
            assert tensor.shape[1] % 4 == 0, "All concatenated tensors must have channels as multiples of 4."
        
        return torch.cat(x, dim=self.dim)

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
