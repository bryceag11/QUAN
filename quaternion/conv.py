# quaternion/conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional
import math
from .qbatch_norm import QBN, IQBN
__all__ = ['QConv', 'QConv1D', 'QConv2D',
           'QConv3D', 'QDense', 'QInit']



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# def autopad(k: int, p: Optional[int] = None) -> int:
#     """Automatic padding calculation based on kernel size.
    
#     Args:
#         k (int): Kernel size.
#         p (int, optional): Desired padding. If None, calculates 'same' padding.
        
#     Returns:
#         int: Padding size.
#     """
#     if p is None:
#         return (k - 1) // 2
#     return p

class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = QConv2D(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = IQBN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

# Geometric
# class QConv(nn.Module):
#     def __init__(self, 
#                  rank: int,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Union[int, Tuple[int, ...]],
#                  stride: Union[int, Tuple[int, ...]] = 1,
#                  padding: Union[str, int, Tuple[int, ...]] = 0,
#                  dilation: Union[int, Tuple[int, ...]] = 1,
#                  groups: int = 1,
#                  bias: bool = True,
#                  padding_mode: str = 'zeros',
#                  device=None,
#                  dtype=None) -> None:
#         super(QConv, self).__init__()
        
#         self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
#         assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
#         self.rank = rank
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.groups = groups
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
#         if rank == 1:
#             Conv = nn.Conv1d
#         elif rank == 2:
#             Conv = nn.Conv2d
#         else:
#             Conv = nn.Conv3d
            
#         self.is_first_layer = (in_channels == 3)
        
#         if self.is_first_layer:
#             actual_in_channels = 1
#         else:
#             assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
#             actual_in_channels = in_channels // 4
            
#         assert out_channels % 4 == 0, "out_channels must be multiple of 4"
#         out_channels_quat = out_channels // 4
        
#         # Create separate convolutions for each geometric component
#         self.conv_r = Conv(actual_in_channels, out_channels_quat, kernel_size,
#                           stride, padding, dilation, groups, bias, 
#                           padding_mode, device=self.device, dtype=dtype)
        
#         self.conv_i = Conv(actual_in_channels, out_channels_quat, kernel_size,
#                           stride, padding, dilation, groups, False, 
#                           padding_mode, device=self.device, dtype=dtype)
        
#         self.conv_j = Conv(actual_in_channels, out_channels_quat, kernel_size,
#                           stride, padding, dilation, groups, False, 
#                           padding_mode, device=self.device, dtype=dtype)
        
#         self.conv_k = Conv(actual_in_channels, out_channels_quat, kernel_size,
#                           stride, padding, dilation, groups, False, 
#                           padding_mode, device=self.device, dtype=dtype)
        
#         # Learnable rotation parameters
#         self.theta = nn.Parameter(torch.Tensor(out_channels_quat))
#         self.axis_x = nn.Parameter(torch.Tensor(out_channels_quat))
#         self.axis_y = nn.Parameter(torch.Tensor(out_channels_quat))
#         self.axis_z = nn.Parameter(torch.Tensor(out_channels_quat))
        
#         self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.to(self.device)
        
#         if x.size(1) == 3:
#             x = self.rgb_to_quaternion(x)
        
#         if self.is_first_layer:
#             B, Q, H, W = x.shape
#             assert Q == 4, "First layer input must have 4 quaternion components"
            
#             # Process each component independently
#             r_conv = self.conv_r(x[:, 0:1])
#             i_conv = self.conv_i(x[:, 1:2])
#             j_conv = self.conv_j(x[:, 2:3])
#             k_conv = self.conv_k(x[:, 3:4])
#         else:
#             # For subsequent layers, input shape is [B, C, Q, H, W]
#             x_r = x[:, :, 0, :, :]  # shape: [B, C_q, H, W]
#             x_i = x[:, :, 1, :, :]
#             x_j = x[:, :, 2, :, :]
#             x_k = x[:, :, 3, :, :]
            
#             # Independent convolutions
#             r_conv = self.conv_r(x_r)
#             i_conv = self.conv_i(x_i)
#             j_conv = self.conv_j(x_j)
#             k_conv = self.conv_k(x_k)
        
#         # Get output shape
#         B, C, H, W = r_conv.shape
        
#         # Normalize rotation axis
#         axis_norm = torch.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
#         u_x = (self.axis_x / axis_norm).view(1, -1, 1, 1)
#         u_y = (self.axis_y / axis_norm).view(1, -1, 1, 1)
#         u_z = (self.axis_z / axis_norm).view(1, -1, 1, 1)
        
#         # Compute rotation quaternion components
#         theta = self.theta.view(1, -1, 1, 1)
#         cos_half_theta = torch.cos(theta / 2)
#         sin_half_theta = torch.sin(theta / 2)
        
#         # Apply geometric transformation
#         # First rotation
#         out_r = cos_half_theta * r_conv - sin_half_theta * (u_x * i_conv + u_y * j_conv + u_z * k_conv)
#         out_i = cos_half_theta * i_conv + sin_half_theta * (u_x * r_conv + u_y * k_conv - u_z * j_conv)
#         out_j = cos_half_theta * j_conv + sin_half_theta * (u_y * r_conv - u_x * k_conv + u_z * i_conv)
#         out_k = cos_half_theta * k_conv + sin_half_theta * (u_z * r_conv + u_x * j_conv - u_y * i_conv)
        
#         # Stack outputs
#         out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
#         # Clean up intermediate tensors
#         del r_conv, i_conv, j_conv, k_conv
        
#         return out

#     def rgb_to_quaternion(self, x):
#         """Convert RGB input to quaternion format [B, 4, H, W]"""
#         rgb_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8)
        
#         return torch.cat([
#             rgb_normalized.mean(dim=1, keepdim=True),  # real part
#             rgb_normalized[:, 0:1],  # R -> i
#             rgb_normalized[:, 1:2],  # G -> j
#             rgb_normalized[:, 2:3]   # B -> k
#         ], dim=1)

#     def _initialize_weights(self):
#         # Calculate fan_in
#         kernel_prod = np.prod(self.kernel_size)
#         fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
#         # Initialize convolution weights
#         nn.init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
#         if self.conv_r.bias is not None:
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.conv_r.bias, -bound, bound)
        
#         # Initialize imaginary convolutions with smaller weights
#         scale = 0.5
#         for conv in [self.conv_i, self.conv_j, self.conv_k]:
#             nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scale)
        
#         # Initialize geometric parameters
#         nn.init.uniform_(self.theta, -math.pi/4, math.pi/4)
#         nn.init.uniform_(self.axis_x, -1, 1)
#         nn.init.uniform_(self.axis_y, -1, 1)
#         nn.init.uniform_(self.axis_z, -1, 1)
        
#         # Normalize rotation axis
#         with torch.no_grad():
#             norm = torch.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
#             self.axis_x.div_(norm)
#             self.axis_y.div_(norm)
#             self.axis_z.div_(norm)

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
                 dtype=None,
                 mapping_type: str = 'raw_normalized') -> None:
        super(QConv, self).__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized', 'hamilton', 'poincare']
        assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
        self.mapping_type = mapping_type
        # Special handling for first layer

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
        self.mapping_type = mapping_type
        # Define the underlying real-valued convolution for each quaternion component
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        self.is_first_layer = (in_channels == 3)  # Changed from 4 to 3
        if self.is_first_layer:
            # For RGB input, map to 4 channels
            actual_in_channels = 1  # Use this for the convolution
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        
        # For first layer, use in_channels=1, for others use in_channels//4
        out_channels_quat = out_channels // 4
        
        self.conv_r = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode, device=self.device, dtype=dtype).to(self.device)
        
        self.conv_i = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode, device=self.device, dtype=dtype).to(self.device)
        
        self.conv_j = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode, device=self.device, dtype=dtype).to(self.device)
        
        self.conv_k = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode, device=self.device, dtype=dtype).to(self.device)
                          
        self._initialize_weights()

    def _initialize_weights(self):
        # Calculate fan_in for initialization
        kernel_prod = np.prod(self.kernel_size)
        
        # Handle first layer vs. subsequent layers
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Initialize real convolution (conv_rr)
        nn.init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
        if self.conv_r.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv_r.bias, -bound, bound)
        
        # Initialize imaginary parts with smaller weights
        scale_factors = {
            'luminance': [1.0, 0.5, 0.5, 0.5],      # Emphasize luminance
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],  # Slightly more balanced
            'raw_normalized': [1.0, 1.0, 1.0, 1.0]  # Equal emphasis
        }
        
        scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
        convs = [self.conv_i, self.conv_j, self.conv_k]
        for i, conv in enumerate(convs):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scales[i+1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        
        # Handle RGB input conversion for first layer
        if x.size(1) == 3:  # RGB input
            with torch.no_grad():  # Prevent memory accumulation during conversion
                x = self.rgb_to_quaternion(x)
        
        if self.is_first_layer:
            B, Q, H, W = x.shape
            assert Q == 4, "First layer input must have 4 quaternion components"
            
            # Process each component independently
            with torch.set_grad_enabled(self.training):
                r_conv = self.conv_r(x[:, 0:1])
                i_conv = self.conv_i(x[:, 1:2])
                j_conv = self.conv_j(x[:, 2:3])
                k_conv = self.conv_k(x[:, 3:4])
        else:
            # For subsequent layers, input shape is [B, C, Q, H, W]
            x_r = x[:, :, 0, :, :]  # shape: [B, C_q, H, W]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]
            
            # Independent convolutions for each component
            with torch.set_grad_enabled(self.training):
                r_conv = self.conv_r(x_r)
                i_conv = self.conv_i(x_i)
                j_conv = self.conv_j(x_j)
                k_conv = self.conv_k(x_k)
        
        # Hamilton product mixing - same for both cases
        # Use in-place operations where possible
        out_r = r_conv
        out_r.sub_(i_conv)
        out_r.sub_(j_conv)
        out_r.sub_(k_conv)
        
        out_i = r_conv.clone()
        out_i.add_(i_conv)
        out_i.add_(j_conv)
        out_i.sub_(k_conv)
        
        out_j = r_conv.clone()
        out_j.sub_(i_conv)
        out_j.add_(j_conv)
        out_j.add_(k_conv)
        
        out_k = r_conv.clone()
        out_k.add_(i_conv)
        out_k.sub_(j_conv)
        out_k.add_(k_conv)
        
        # Stack outputs
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        # Clean up intermediate tensors
        del r_conv, i_conv, j_conv, k_conv
        
        return out

    def rgb_to_quaternion(self, rgb_input):
        """Convert RGB to quaternion-like representation"""
        B, C, H, W = rgb_input.shape
        
        # Luminance-based
        luminance = (0.299 * rgb_input[:, 0] + 
                     0.587 * rgb_input[:, 1] + 
                     0.114 * rgb_input[:, 2]).unsqueeze(1)
        
        # Mean Brightness
        mean_brightness = rgb_input.mean(dim=1, keepdim=True)
        
        # Normalized Channels
        rgb_normalized = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())
        
        # New Hamilton mapping
        def hamilton_mapping(x):
            # Map RGB to pure quaternion using Hamilton embedding
            # Real part is 0, and RGB maps to i,j,k components
            real = torch.zeros_like(x[:, 0:1])
            return torch.cat([
                real,  # Real part (0)
                x[:, 0:1],  # R -> i component
                x[:, 1:2],  # G -> j component
                x[:, 2:3]   # B -> k component
            ], dim=1)
        
        # New Poincaré mapping
        def poincare_mapping(x):
            # Normalize to unit ball (Poincaré disk)
            norm = torch.norm(x, dim=1, keepdim=True)
            x_normalized = x / (norm + 1)  # Map to Poincaré ball
            
            # Create quaternion with normalized RGB
            return torch.cat([
                torch.sqrt(1 - torch.sum(x_normalized**2, dim=1, keepdim=True)),  # Real part
                x_normalized[:, 0:1],  # R component
                x_normalized[:, 1:2],  # G component
                x_normalized[:, 2:3]   # B component
            ], dim=1)
    
        # Mapping strategies
        mappings = {
            'luminance': torch.cat([luminance, 
                                     rgb_normalized[:, 0:1], 
                                     rgb_normalized[:, 1:2], 
                                     rgb_normalized[:, 2:3]], dim=1),
            
            'mean_brightness': torch.cat([mean_brightness, 
                                           rgb_input[:, 0:1], 
                                           rgb_input[:, 1:2], 
                                           rgb_input[:, 2:3]], dim=1),
            
            'raw_normalized': torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
                                          rgb_normalized[:, 0:1], 
                                          rgb_normalized[:, 1:2], 
                                          rgb_normalized[:, 2:3]], dim=1),
            'hamilton': hamilton_mapping(rgb_input),
            'poincare': poincare_mapping(rgb_input)        
        }
        
        return mappings[self.mapping_type]

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


class QDense(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 mapping_type: str = 'luminance',
                 device=None,
                 dtype=None):
        super(QDense, self).__init__()

                # Add mapping strategy
        self.mapping_type = mapping_type
        
        # Ensure input features are handled correctly
        if in_features == 3:  # If input is RGB
            # Adjust input features after mapping
            in_features = 4
        else:
            # Ensure input features are a multiple of 4
            assert in_features % 4 == 0, "in_features must be a multiple of 4"
        
        assert out_features % 4 == 0, "out_features must be a multiple of 4"
        # Compute feature dimensions
        in_features_quat = in_features // 4
        out_features_quat = out_features // 4
        
        # Create separate linear layers for each quaternion component
        self.linear_rr = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        self.linear_ri = nn.Linear(in_features_quat, out_features_quat, bias=False)
        self.linear_rj = nn.Linear(in_features_quat, out_features_quat, bias=False)
        self.linear_rk = nn.Linear(in_features_quat, out_features_quat, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def rgb_to_quaternion(self, rgb_input):
        """Convert RGB to quaternion-like representation"""
        B, C = rgb_input.shape
        
        # Luminance-based
        luminance = (0.299 * rgb_input[:, 0] + 
                     0.587 * rgb_input[:, 1] + 
                     0.114 * rgb_input[:, 2]).unsqueeze(1)
        
        # Mean Brightness
        mean_brightness = rgb_input.mean(dim=1, keepdim=True)
        
        # Normalized Channels
        rgb_normalized = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())
        
        # Mapping strategies
        mappings = {
            'luminance': torch.cat([luminance, 
                                     rgb_normalized[:, 0:1], 
                                     rgb_normalized[:, 1:2], 
                                     rgb_normalized[:, 2:3]], dim=1),
            
            'mean_brightness': torch.cat([mean_brightness, 
                                           rgb_input[:, 0:1], 
                                           rgb_input[:, 1:2], 
                                           rgb_input[:, 2:3]], dim=1),
            
            'raw_normalized': torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
                                          rgb_normalized[:, 0:1], 
                                          rgb_normalized[:, 1:2], 
                                          rgb_normalized[:, 2:3]], dim=1)
        }
        
        return mappings[self.mapping_type]
    
    def _initialize_weights(self):
        # Similar to convolution initialization
        for linear in [self.linear_rr, self.linear_ri, self.linear_rj, self.linear_rk]:
            nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
            if linear.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(linear.bias, -bound, bound)
        
        # Scale down weights for imaginary components
        for linear in [self.linear_ri, self.linear_rj, self.linear_rk]:
            linear.weight.data *= 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            x = self.rgb_to_quaternion(x)
        # Separate input into quaternion components
        x_r = x[:, :x.size(1)//4]
        x_i = x[:, x.size(1)//4:x.size(1)//2]
        x_j = x[:, x.size(1)//2:3*x.size(1)//4]
        x_k = x[:, 3*x.size(1)//4:]
        
        # Apply linear transformations with Hamilton product rules
        r_r = self.linear_rr(x_r)
        r_i = self.linear_ri(x_r)
        r_j = self.linear_rj(x_r)
        r_k = self.linear_rk(x_r)
        
        i_r = self.linear_rr(x_i)
        i_i = self.linear_ri(x_i)
        i_j = self.linear_rj(x_i)
        i_k = self.linear_rk(x_i)
        
        j_r = self.linear_rr(x_j)
        j_i = self.linear_ri(x_j)
        j_j = self.linear_rj(x_j)
        j_k = self.linear_rk(x_j)
        
        k_r = self.linear_rr(x_k)
        k_i = self.linear_ri(x_k)
        k_j = self.linear_rj(x_k)
        k_k = self.linear_rk(x_k)
        
        # Hamilton product rules for output
        out_r = r_r - i_i - j_j - k_k
        out_i = r_i + i_r + j_k - k_j
        out_j = r_j - i_k + j_r + k_i
        out_k = r_k + i_j - j_i + k_r
        
        # Stack back into quaternion format
        out = torch.stack([out_r, out_i, out_j, out_k], dim=1)
        out = out.view(x.size(0), -1)
        
        return out