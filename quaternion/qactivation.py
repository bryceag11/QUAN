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
    Quaternion ReLU activation function that applies ReLU separately to each quaternion component
    following the paper's split-type activation approach.
    
    The input tensor should be in the format (batch_size, channels, 4, height, width) where:
    - The 4 represents quaternion components in order: real, i, j, k
    """
    def __init__(self, inplace: bool = False):
        super(QReLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a new tensor instead of modifying in place
        out = torch.zeros_like(x)
        
        # Copy the real component unchanged
        out[:,:,0,:,:] = x[:,:,0,:,:]
        
        # Apply ReLU to imaginary components
        out[:,:,1:,:,:] = F.relu(x[:,:,1:,:,:])
        
        return out
    
    def extra_repr(self) -> str:
        return ''



class QPReLU(nn.Module):
    """
    Split Quaternion Parametric ReLU Activation Function.
    Applies PReLU to each quaternion component separately with learnable parameters.
    """
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super(QPReLU, self).__init__()
        self.num_parameters = num_parameters
        
        # Create 4 learnable parameters, one for each quaternion component
        self.weight = nn.Parameter(torch.Tensor(4 * num_parameters).fill_(init))
        
    def forward(self, x):
        # x shape: (batch_size, channels, 4, height, width)
        # Get parameters for each component
        w_r, w_i, w_j, w_k = self.weight.chunk(4)
        
        # Apply component-wise PReLU
        x_r = F.prelu(x[:,:,0,:,:], w_r)
        x_i = F.prelu(x[:,:,1,:,:], w_i)
        x_j = F.prelu(x[:,:,2,:,:], w_j) 
        x_k = F.prelu(x[:,:,3,:,:], w_k)
        
        # Stack components back together
        return torch.stack([x_r, x_i, x_j, x_k], dim=2)

class QREReLU(nn.Module):
    """
    Quaternion Rotation-Equivariant ReLU Activation Function.
    Preserves the rotation-equivariant properties of quaternions.
    """
    def __init__(self, eps=1e-8):
        """
        Initialize QREReLU activation.
        
        Args:
            eps (float): Small constant to avoid division by zero
        """
        super(QREReLU, self).__init__()
        self.eps = eps

    def forward(self, q):
        # Compute norm of each quaternion
        # q shape: (batch_size, channels, 4, height, width)
        norm = torch.norm(q, dim=2, keepdim=True)
        
        # Compute average norm across batch and spatial dimensions
        avg_norm = torch.mean(norm, dim=(0, 3, 4), keepdim=True)
        
        # Avoid division by zero
        norm = torch.clamp(norm, min=self.eps)
        
        # Apply QREReLU formula: qs * ||qs|| / max(||qs||, c)
        # where c is the average norm
        scale = norm / torch.max(norm, avg_norm)
        return q * scale

    
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


