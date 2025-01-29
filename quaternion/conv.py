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
                 dtype=None,
                 mapping_type: str = 'luminance') -> None:
        super(QConv, self).__init__()
        
        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized']
        assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
        self.mapping_type = mapping_type
        # Special handling for first layer

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized']
        assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
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
        
        self.conv_r = Conv(
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
        
        self.conv_i = Conv(
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
        
        self.conv_j = Conv(
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
        
        self.conv_k = Conv(
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
        """Base forward pass implementation."""
        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)
        if self.is_first_layer:
            B, Q, H, W = x.shape
            assert Q == 4, "First layer input must have 4 quaternion components"
            
            # Process each component independently
            r_conv = self.conv_r(x[:, 0:1])  # [B, C_out/4, H, W]
            i_conv = self.conv_i(x[:, 1:2])
            j_conv = self.conv_j(x[:, 2:3])
            k_conv = self.conv_k(x[:, 3:4])
            
        else:
            x_r = x[:, :, 0, :, :]  # shape: [B, C_q, H, W]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]
            
            # Independent convolutions
            r_conv = self.conv_r(x_r)
            i_conv = self.conv_i(x_i)
            j_conv = self.conv_j(x_j)
            k_conv = self.conv_k(x_k)
        
        # Mix using Hamilton product - single mixing step
        out_r = r_conv - i_conv - j_conv - k_conv
        out_i = r_conv + i_conv + j_conv - k_conv
        out_j = r_conv - i_conv + j_conv + k_conv
        out_k = r_conv + i_conv - j_conv + k_conv
        
        # Stack outputs
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
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

# class QConv(nn.Module):
#     """
#     Base Quaternion Convolution class.
#     """
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
#                  dtype=None,
#                  mapping_type: str = 'luminance') -> None:
#         super(QConv, self).__init__()
        
#         assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
#         valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized']
#         assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
#         self.mapping_type = mapping_type
#         # Special handling for first layer

#         self.rank = rank
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.groups = groups
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
#         valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized']
#         assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
#         self.mapping_type = mapping_type
#         # Define the underlying real-valued convolution for each quaternion component
#         if rank == 1:
#             Conv = nn.Conv1d
#         elif rank == 2:
#             Conv = nn.Conv2d
#         else:
#             Conv = nn.Conv3d
            
#         self.is_first_layer = (in_channels == 3)  # Changed from 4 to 3
#         if self.is_first_layer:
#             # For RGB input, map to 4 channels
#             actual_in_channels = 1  # Use this for the convolution
#         else:
#             assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
#             actual_in_channels = in_channels // 4
#         assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        
#         # For first layer, use in_channels=1, for others use in_channels//4
#         out_channels_quat = out_channels // 4
        
#         self.conv_rr = Conv(
#             actual_in_channels,
#             out_channels_quat,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype
#         )
        
#         self.conv_ri = Conv(
#             actual_in_channels,
#             out_channels_quat,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=False,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype
#         )
        
#         self.conv_rj = Conv(
#             actual_in_channels,
#             out_channels_quat,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=False,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype
#         )
        
#         self.conv_rk = Conv(
#             actual_in_channels,
#             out_channels_quat,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=False,
#             padding_mode=padding_mode,
#             device=device,
#             dtype=dtype
#         )
        
#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self):
#         # Calculate fan_in for initialization
#         kernel_prod = np.prod(self.kernel_size)
        
#         # Handle first layer vs. subsequent layers
#         fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
#         # Initialize real convolution (conv_rr)
#         nn.init.kaiming_uniform_(self.conv_rr.weight, a=math.sqrt(5))
#         if self.conv_rr.bias is not None:
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.conv_rr.bias, -bound, bound)
        
#         # Initialize imaginary parts with smaller weights
#         scale_factors = {
#             'luminance': [1.0, 0.5, 0.5, 0.5],      # Emphasize luminance
#             'mean_brightness': [1.0, 0.75, 0.75, 0.75],  # Slightly more balanced
#             'raw_normalized': [1.0, 1.0, 1.0, 1.0]  # Equal emphasis
#         }
        
#         scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
#         convs = [self.conv_ri, self.conv_rj, self.conv_rk]
#         for i, conv in enumerate(convs):
#             nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scales[i+1])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Base forward pass implementation."""
#         if x.size(1) == 3:  # RGB input
#             x = self.rgb_to_quaternion(x)
#         if self.is_first_layer:  # First layer
#             B, Q, H, W = x.shape
#             assert Q == 4, "First layer input must have 4 quaternion components"
            
#             # Process each quaternion component separately
#             components = []
#             for i in range(4):
#                 comp = x[:, i:i+1]  # Keep dim: [B, 1, H, W]
#                 components.append(comp)
            
#             # Apply convolutions with Hamilton product
#             out_r = self.conv_rr(components[0]) - self.conv_ri(components[1]) - self.conv_rj(components[2]) - self.conv_rk(components[3])
#             out_i = self.conv_rr(components[1]) + self.conv_ri(components[0]) + self.conv_rj(components[3]) - self.conv_rk(components[2])
#             out_j = self.conv_rr(components[2]) - self.conv_ri(components[3]) + self.conv_rj(components[0]) + self.conv_rk(components[1])
#             out_k = self.conv_rr(components[3]) + self.conv_ri(components[2]) - self.conv_rj(components[1]) + self.conv_rk(components[0])
            
#             # Stack outputs
#             out = torch.stack([out_r, out_i, out_j, out_k], dim=2)  # [B, C_out//4, 4, H, W]
            
#             return out
            
#         else:  # Later layers
#             x_r = x[:, :, 0, :, :]  # shape: [B, C_q, H, W]
#             x_i = x[:, :, 1, :, :]
#             x_j = x[:, :, 2, :, :]
#             x_k = x[:, :, 3, :, :]


#             # Apply quaternion convolutions
#             r_r = self.conv_rr(x_r) # [B, C_out/4, H, W]
#             r_i = self.conv_ri(x_r)
#             r_j = self.conv_rj(x_r)
#             r_k = self.conv_rk(x_r)

#             i_r = self.conv_rr(x_i)
#             i_i = self.conv_ri(x_i)
#             i_j = self.conv_rj(x_i)
#             i_k = self.conv_rk(x_i)

#             j_r = self.conv_rr(x_j)
#             j_i = self.conv_ri(x_j)
#             j_j = self.conv_rj(x_j)
#             j_k = self.conv_rk(x_j)

#             k_r = self.conv_rr(x_k)
#             k_i = self.conv_ri(x_k)
#             k_j = self.conv_rj(x_k)
#             k_k = self.conv_rk(x_k)

#             # Hamilton product
#             out_r = r_r - i_i - j_j - k_k
#             out_i = r_i + i_r + j_k - k_j
#             out_j = r_j - i_k + j_r + k_i
#             out_k = r_k + i_j - j_i + k_r

#             # Stack back into quaternion format
#             # Result shape: [B, C_out/4, 4, H, W]
#             out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
#             return out

#     def rgb_to_quaternion(self, rgb_input):
#         """Convert RGB to quaternion-like representation"""
#         B, C, H, W = rgb_input.shape
        
#         # Luminance-based
#         luminance = (0.299 * rgb_input[:, 0] + 
#                      0.587 * rgb_input[:, 1] + 
#                      0.114 * rgb_input[:, 2]).unsqueeze(1)
        
#         # Mean Brightness
#         mean_brightness = rgb_input.mean(dim=1, keepdim=True)
        
#         # Normalized Channels
#         rgb_normalized = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())
        
#         # Mapping strategies
#         mappings = {
#             'luminance': torch.cat([luminance, 
#                                      rgb_normalized[:, 0:1], 
#                                      rgb_normalized[:, 1:2], 
#                                      rgb_normalized[:, 2:3]], dim=1),
            
#             'mean_brightness': torch.cat([mean_brightness, 
#                                            rgb_input[:, 0:1], 
#                                            rgb_input[:, 1:2], 
#                                            rgb_input[:, 2:3]], dim=1),
            
#             'raw_normalized': torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
#                                           rgb_normalized[:, 0:1], 
#                                           rgb_normalized[:, 1:2], 
#                                           rgb_normalized[:, 2:3]], dim=1)
#         }
        
#         return mappings[self.mapping_type]

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
                 dtype=None,
                 mapping_type: str = 'luminance') -> None:
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
            dtype=dtype,
            mapping_type=mapping_type
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