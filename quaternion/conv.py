# quaternion/conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math

__all__ = ['QConv', 'QConv1D', 'QConv2D',
           'QConv3D', 'QDense', 'QInit']
class QConv(nn.Module):
    """
    Base Quaternion Convolution class.
    """
    def __init__(self, 
                 rank: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv, self).__init__()
        
        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        # Special handling for first layer
        self.is_first_layer = (in_channels == 4)
        if not self.is_first_layer:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 (except for first layer)"
            
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
        # Define the underlying real-valued convolution for each quaternion component
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        # For first layer, use in_channels=1, for others use in_channels//4
        actual_in_channels = 1 if self.is_first_layer else in_channels // 4
        out_channels_quat = out_channels // 4
        
        self.conv_rr = Conv(
            actual_in_channels,
            out_channels_quat,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        
        self.conv_ri = Conv(
            actual_in_channels,
            out_channels_quat,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        
        self.conv_rj = Conv(
            actual_in_channels,
            out_channels_quat,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        
        self.conv_rk = Conv(
            actual_in_channels,
            out_channels_quat,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Calculate fan_in for initialization
        kernel_prod = np.prod(self.kernel_size)
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Initialize conv_rr
        nn.init.kaiming_uniform_(self.conv_rr.weight, a=math.sqrt(5))
        if self.conv_rr.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv_rr.bias, -bound, bound)
        
        # Initialize other convolutions
        for conv in [self.conv_ri, self.conv_rj, self.conv_rk]:
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward pass implementation."""
        if self.is_first_layer and x.dim() == 4:  # First layer
            B, Q, H, W = x.shape
            assert Q == 4, "First layer input must have 4 quaternion components"
            
            # Process each quaternion component separately
            components = []
            for i in range(4):
                comp = x[:, i:i+1]  # Keep dim: [B, 1, H, W]
                components.append(comp)
            
            # Apply convolutions with Hamilton product
            out_r = self.conv_rr(components[0]) - self.conv_ri(components[1]) - self.conv_rj(components[2]) - self.conv_rk(components[3])
            out_i = self.conv_rr(components[1]) + self.conv_ri(components[0]) + self.conv_rj(components[3]) - self.conv_rk(components[2])
            out_j = self.conv_rr(components[2]) - self.conv_ri(components[3]) + self.conv_rj(components[0]) + self.conv_rk(components[1])
            out_k = self.conv_rr(components[3]) + self.conv_ri(components[2]) - self.conv_rj(components[1]) + self.conv_rk(components[0])
            
            # Stack outputs
            out = torch.stack([out_r, out_i, out_j, out_k], dim=2)  # [B, C_out//4, 4, H, W]
            
            return out
            
        else:  # Later layers
            x_r = x[:, :, 0, :, :]  # shape: [B, C_q, H, W]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]

            # Apply quaternion convolutions
            r_r = self.conv_rr(x_r) # [B, C_out/4, H, W]
            r_i = self.conv_ri(x_r)
            r_j = self.conv_rj(x_r)
            r_k = self.conv_rk(x_r)

            i_r = self.conv_rr(x_i)
            i_i = self.conv_ri(x_i)
            i_j = self.conv_rj(x_i)
            i_k = self.conv_rk(x_i)

            j_r = self.conv_rr(x_j)
            j_i = self.conv_ri(x_j)
            j_j = self.conv_rj(x_j)
            j_k = self.conv_rk(x_j)

            k_r = self.conv_rr(x_k)
            k_i = self.conv_ri(x_k)
            k_j = self.conv_rj(x_k)
            k_k = self.conv_rk(x_k)

            # Hamilton product
            out_r = r_r - i_i - j_j - k_k
            out_i = r_i + i_r + j_k - k_j
            out_j = r_j - i_k + j_r + k_i
            out_k = r_k + i_j - j_i + k_r

            # Stack back into quaternion format
            # Result shape: [B, C_out/4, 4, H, W]
            out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
            return out

class QConv1D(QConv):
    """1D Quaternion Convolution layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[str, int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv1D, self).__init__(
            rank=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )



class QConv2D(QConv):
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
        super().__init__(
            rank=2,  # Fixed for 2D convolution
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )


class QConv3D(QConv):
    """3D Quaternion Convolution layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv3D, self).__init__(
            rank=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )