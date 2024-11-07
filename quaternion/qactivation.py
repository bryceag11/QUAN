# Quaternion activations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math 



__all__ = ['QHardTanh', 'QLeakyReLU', 'QuaternionActivation', 'QReLU', 'QPReLU', 'QREReLU', 'QSigmoid', 'QTanh']

class QuaternionActivation(nn.Module):
    """
    Quaternion Activation Function.
    Applies a real-valued activation function to each quaternion component.
    """
    def __init__(self, activation=nn.SiLU()):
        super(QuaternionActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        # Apply activation to each component: [xR, xI, xJ, xK]
        return self.activation(x)
    
class QSigmoid(nn.Module):
    """
    Split Quaternion Sigmoid Activation Function.
    Applies sigmoid to each quaternion component separately.
    """
    def __init__(self):
        super(QSigmoid, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, q):
        # q shape: (batch_size, channels, 4, ...)
        return self.sigmoid(q)

class QTanh(nn.Module):
    """
    Split Quaternion Hyperbolic Tangent Activation Function.
    Applies tanh to each quaternion component separately.
    """
    def __init__(self):
        super(QTanh, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, q):
        return self.tanh(q)

class QHardTanh(nn.Module):
    """
    Split Quaternion Hard Hyperbolic Tangent Activation Function.
    Applies hardtanh to each quaternion component separately.
    """
    def __init__(self, min_val=-1.0, max_val=1.0):
        super(QHardTanh, self).__init__()
        self.hardtanh = nn.Hardtanh(min_val, max_val)

    def forward(self, q):
        return self.hardtanh(q)
    
class QReLU(nn.Module):
    """
    Split Quaternion ReLU Activation Function.
    Applies ReLU to each quaternion component separately.
    """
    def __init__(self):
        super(QReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, q):
        return self.relu(q)



class QPReLU(nn.Module):
    """
    Split Quaternion Parametric ReLU Activation Function.
    Applies PReLU to each quaternion component separately.
    """
    def __init__(self, num_parameters=4):
        """
        Initializes the QPReLU activation.

        Args:
            num_parameters (int): Number of PReLU parameters. Should match the quaternion dimension (4).
        """
        super(QPReLU, self).__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, q):
        return self.prelu(q)
    
class QLeakyReLU(nn.Module):
    """
    Split Quaternion Leaky ReLU Activation Function.
    Applies Leaky ReLU to each quaternion component separately.
    """
    def __init__(self, negative_slope=0.01):
        super(QLeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, q):
        return self.leaky_relu(q)
    
class QREReLU(nn.Module):
    """
    Quaternion Rotation-Equivariant ReLU Activation Function.
    Preserves the rotation-equivariant properties of quaternions.
    """
    def __init__(self, c=1.0, eps=1e-8):
        """
        Initializes the QREReLU activation.

        Args:
            c (float): Scaling constant.
            eps (float): Small constant to avoid division by zero.
        """
        super(QREReLU, self).__init__()
        self.c = c
        self.eps = eps

    def forward(self, q):
        # Compute norm of each quaternion
        norm = torch.norm(q, dim=2, keepdim=True)  # Shape: (batch_size, channels, 1, ...)
        # Compute average norm
        avg_norm = torch.mean(norm, dim=(0, 3, 4), keepdim=True)  # Adjust dimensions as needed
        # Compute c as per definition
        c = avg_norm

        # Avoid division by zero
        norm_clamped = torch.clamp(norm, min=self.eps)

        # Apply the QREReLU formula
        factor = norm_clamped / torch.max(norm_clamped, c)
        return factor * q


