#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

class QDense(nn.Module):
    """
    Quaternion Dense (fully connected) layer.
    """
    def __init__(self,
                 in_features: int,
                 units: int,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        """
        Initialize quaternion dense layer.
        
        Args:
            in_features: Number of input features (should be divisible by 3)
            units: Number of output units (will be multiplied by 3 internally)
            bias: Whether to include bias
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(QDense, self).__init__()
        
        assert in_features % 3 == 0, "Input features must be divisible by 3"
        self.in_features = in_features // 3
        self.units = units

        # Initialize phase kernel using normal distribution
        kernel_shape = (self.in_features, self.units)
        phase = torch.empty(kernel_shape, **factory_kwargs)
        self.phase_kernel = nn.Parameter(torch.nn.init.normal_(phase, 0, np.pi/2))

        # Initialize modulus kernel using normal distribution
        fan_in = self.in_features
        s = np.sqrt(1. / fan_in)
        modulus = torch.empty(kernel_shape, **factory_kwargs)
        self.modulus_kernel = nn.Parameter(torch.nn.init.normal_(modulus, 0, s))

        if bias:
            self.bias = nn.Parameter(torch.zeros(3 * units, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing quaternion matrix multiplication.
        """
        # Calculate quaternion components
        f_phase1 = torch.cos(self.phase_kernel)
        f_phase2 = torch.sin(self.phase_kernel) * (3**0.5/3)

        # Calculate filter components
        f1 = (torch.pow(f_phase1, 2) - torch.pow(f_phase2, 2)) * self.modulus_kernel
        f2 = (2 * (torch.pow(f_phase2, 2) - f_phase2 * f_phase1)) * self.modulus_kernel
        f3 = (2 * (torch.pow(f_phase2, 2) + f_phase2 * f_phase1)) * self.modulus_kernel
        f4 = (2 * (torch.pow(f_phase2, 2) + f_phase2 * f_phase1)) * self.modulus_kernel
        f5 = (torch.pow(f_phase1, 2) - torch.pow(f_phase2, 2)) * self.modulus_kernel
        f6 = (2 * (torch.pow(f_phase2, 2) - f_phase2 * f_phase1)) * self.modulus_kernel
        f7 = (2 * (torch.pow(f_phase2, 2) - f_phase2 * f_phase1)) * self.modulus_kernel
        f8 = (2 * (torch.pow(f_phase2, 2) + f_phase2 * f_phase1)) * self.modulus_kernel
        f9 = (torch.pow(f_phase1, 2) - torch.pow(f_phase2, 2)) * self.modulus_kernel

        # Construct transformation matrix
        matrix1 = torch.cat([f1, f2, f3], dim=1)
        matrix2 = torch.cat([f4, f5, f6], dim=1)
        matrix3 = torch.cat([f7, f8, f9], dim=1)
        matrix = torch.cat([matrix1, matrix2, matrix3], dim=0)

        # Apply transformation
        output = F.linear(input, matrix.t(), self.bias)
        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return f'in_features={self.in_features * 3}, out_features={self.units * 3}, bias={self.bias is not None}'