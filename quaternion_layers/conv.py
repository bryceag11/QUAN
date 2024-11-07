#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math

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
        
        # Initialize phase weights
        phase = torch.empty(kernel_shape, device=device, dtype=dtype)
        self.phase = nn.Parameter(torch.nn.init.uniform_(phase, -np.pi, np.pi))

        # Initialize modulus weights
        fan_in = self.in_channels * np.prod(self.kernel_size)
        modulus = torch.empty(kernel_shape, device=device, dtype=dtype)
        bound = 1 / math.sqrt(fan_in)
        self.modulus = nn.Parameter(torch.nn.init.uniform_(modulus, -bound, bound))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def _normalize_tuple(self, value, rank, name):
        if isinstance(value, int):
            return (value,) * rank
        else:
            return tuple(value)

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
        padding = self._get_padding(input.shape)
        
        if isinstance(padding, tuple) and len(padding) > 2:
            input = F.pad(input, padding, mode=self.padding_mode, value=0)
            padding = 0

        # Get quaternion components
        cos_weights = torch.cos(self.phase)
        sin_weights = torch.sin(self.phase)

        # Create quaternion weights
        real_weight = cos_weights * self.modulus
        imag_weight = sin_weights * self.modulus

        # Perform convolution
        if self.rank == 1:
            conv_fn = F.conv1d
        elif self.rank == 2:
            conv_fn = F.conv2d
        else:  # rank == 3
            conv_fn = F.conv3d

        # Convolve input with real and imaginary weights
        output_real = conv_fn(input, real_weight, None, self.strides, padding, self.dilation, self.groups)
        output_imag = conv_fn(input, imag_weight, None, self.strides, padding, self.dilation, self.groups)

        # Combine outputs according to quaternion algebra
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