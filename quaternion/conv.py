#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math

__all__ = ['QConv', 'QConv1D', 'QConv2D',
           'QConv3D', 'QDense', 'QInit']

class QConv(nn.Module):
    def __init__(self, 
                 rank: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 strides: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv, self).__init__()
        
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = self._normalize_tuple(strides, rank, 'strides')
        self.dilation = self._normalize_tuple(dilation, rank, 'dilation')
        self.padding = padding
        self.groups = groups
        pad_mode_map = {
            'zeros': 'constant',
            'reflect': 'reflect',
            'replicate': 'replicate',
            'circular': 'circular'
        }
        self.padding_mode = pad_mode_map.get(padding_mode, 'constant')
        
        # Create weight parameters
        kernel_shape = (self.out_channels, self.in_channels) + self.kernel_size

        # Initialize modulus weights
        fan_in = self.in_channels * np.prod(self.kernel_size)
        modulus = torch.empty(kernel_shape, device=device, dtype=dtype)
        bound = 1 / math.sqrt(fan_in)
        self.modulus = nn.Parameter(torch.nn.init.uniform_(modulus, -bound, bound))

        # Initialize phase weights
        phase = torch.empty(kernel_shape, device=device, dtype=dtype)
        self.phase = nn.Parameter(torch.nn.init.uniform_(phase, -np.pi, np.pi))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def _normalize_tuple(value, rank, name):
        """
        Normalize input to a tuple.
        """
        if isinstance(value, int):
            return (value,) * rank
        elif isinstance(value, tuple) and len(value) == rank:
            return value
        else:
            raise ValueError(f"Invalid {name}. Expected an int or a tuple of length {rank}.")

    def _get_padding(self, input_size):
        if isinstance(self.padding, str) and self.padding.lower() == 'same':
            pad = []
            for i in range(self.rank):
                input_dim = input_size[-(i + 2)]
                kernel_dim = self.kernel_size[-(i + 1)]
                stride = self.strides[-(i + 1)]
                dilation = self.dilation[-(i + 1)]
                
                out_dim = (input_dim + stride - 1) // stride
                pad_needed = max(0, (out_dim - 1) * stride + ((kernel_dim - 1) * dilation + 1) - input_dim)
                pad_start = pad_needed // 2
                pad_end = pad_needed - pad_start
                pad = [pad_start, pad_end] + pad
            return tuple(pad)
        elif isinstance(self.padding, (int, tuple)):
            if isinstance(self.padding, int):
                return (self.padding,) * (2 * self.rank)
            return self.padding
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Compute the padding as needed
        padding = self._get_padding(input.shape)
        
        if isinstance(padding, tuple) and len(padding) > 2:
            input = F.pad(input, padding, mode=self.padding_mode, value=0)
            padding = 0

        # Generate quaternion weights
        cos_weights = torch.cos(self.phase)
        sin_weights = torch.sin(self.phase)
        real_weight = cos_weights * self.modulus
        imag_weight = sin_weights * self.modulus

        # Choose convolution function based on rank
        conv_fn = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.rank]

        # Perform quaternion convolution
        output_real = conv_fn(input, real_weight, None, self.strides, padding, self.dilation, self.groups)
        output_imag = conv_fn(input, imag_weight, None, self.strides, padding, self.dilation, self.groups)

        # Combine real and imaginary outputs according to quaternion algebra
        output = output_real - output_imag

        # Add bias if it exists
        if self.bias is not None:
            output += self.bias.view(1, -1, *([1] * (output.dim() - 2)))

        return output

class QConv1d(QConv):
    """1D Quaternion Convolution layer."""
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: Union[str, int] = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv1d, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)

class QConv2d(QConv):
    """2D Quaternion Convolution layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv2d, self).__init__(
            rank=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)


class QConv3d(QConv):
    """3D Quaternion Convolution layer."""
    def __init__(self,
                 filters: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv3d, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)
        
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
    
