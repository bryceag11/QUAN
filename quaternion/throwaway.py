
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List
import math


# OG LEFT CONV
class LQConv(nn.Module):
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

# HYBRID QCONV
class HybridQConv(nn.Module):
    def __init__(self,
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
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        
        # Handle the special case of first layer (RGB input)
        self.is_first_layer = (in_channels == 3)
        
        # For quaternion convolution, we work with quarter of the channels
        self.quat_in_channels = 1 if self.is_first_layer else in_channels // 4
        self.quat_out_channels = out_channels // 4
        
        # Initialize quaternion weights with shape [out_c/4, in_c/4, 4, kernel_h, kernel_w]
        kernel_shape = (self.quat_out_channels, self.quat_in_channels, 4, *self.kernel_size)
        self.weight = nn.Parameter(torch.empty(kernel_shape, device=device, dtype=dtype))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.quat_out_channels, 4, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
            
        self._initialize_weights()

    def _initialize_weights(self):
        # Calculate fan_in and fan_out for quaternion weights
        fan_in = self.quat_in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.quat_out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        # Initialize weights using Glorot/Xavier uniform
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.weight, -bound, bound)
        
        if self.bias is not None:
            bound_bias = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)


    def forward(self, x):
        if self.is_first_layer:
            x = self.rgb_to_quaternion(x)  # [B, 1, 4, H, W]
        
        # Extract input quaternion components
        x_r = x[:, :, 0]  # [B, C_in/4, H, W]
        x_i = x[:, :, 1]
        x_j = x[:, :, 2]
        x_k = x[:, :, 3]
        
        # Extract weight quaternion components
        w_r = self.weight[:, :, 0]  # [C_out/4, C_in/4, H, W]
        w_i = self.weight[:, :, 1]
        w_j = self.weight[:, :, 2]
        w_k = self.weight[:, :, 3]
        
        # Compute quaternion convolution using F.conv2d
        # Real part
        out_r = F.conv2d(x_r, w_r, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_i, w_i, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_j, w_j, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_k, w_k, None, self.stride, self.padding, self.dilation, self.groups)
        
        # i component
        out_i = F.conv2d(x_r, w_i, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_i, w_r, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_j, w_k, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_k, w_j, None, self.stride, self.padding, self.dilation, self.groups)
        
        # j component
        out_j = F.conv2d(x_r, w_j, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_i, w_k, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_j, w_r, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_k, w_i, None, self.stride, self.padding, self.dilation, self.groups)
        
        # k component
        out_k = F.conv2d(x_r, w_k, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_i, w_j, None, self.stride, self.padding, self.dilation, self.groups) - \
                F.conv2d(x_j, w_i, None, self.stride, self.padding, self.dilation, self.groups) + \
                F.conv2d(x_k, w_r, None, self.stride, self.padding, self.dilation, self.groups)
        
        # Stack outputs [B, C_out/4, 4, H, W]
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 4, 1, 1)
        
        return out
    def rgb_to_quaternion(self, x):
        """Convert RGB input to quaternion format"""
        B, C, H, W = x.shape
        assert C == 3, "Expected RGB input (3 channels)"
        
        # Normalize RGB values
        rgb_normalized = (x - x.min()) / (x.max() - x.min())
        
        # Create quaternion representation [B, 1, 4, H, W]
        # Using normalized mean as real part and RGB as imaginary parts
        quat = torch.cat([
            rgb_normalized.mean(dim=1, keepdim=True),  # real part
            rgb_normalized[:, 0:1],  # R -> i
            rgb_normalized[:, 1:2],  # G -> j
            rgb_normalized[:, 2:3]   # B -> k
        ], dim=1)
        
        return quat.unsqueeze(1)  # [B, 1, 4, H, W]


# Geometric Quaternion Conv
class QConvGeometric(nn.Module):
    """
    Geometric (double-sided) Quaternion Convolution implementation following 
    Zhu et al. [181] approach.
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
        super(QConvGeometric, self).__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
        # Define the underlying real-valued convolution for quaternion components
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        # Handle first layer vs subsequent layers
        self.is_first_layer = (in_channels == 3)  # RGB input
        if self.is_first_layer:
            actual_in_channels = 1
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
            
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        out_channels_quat = out_channels // 4
        
        # Create convolution layers for the geometric transform
        self.conv_magnitude = Conv(actual_in_channels, out_channels_quat, kernel_size,
                                 stride, padding, dilation, groups, False, 
                                 padding_mode, device=self.device, dtype=dtype)
        
        self.conv_theta = Conv(actual_in_channels, out_channels_quat, kernel_size,
                             stride, padding, dilation, groups, False, 
                             padding_mode, device=self.device, dtype=dtype)
        
        # Parameters for rotation axis (learned)
        self.axis_x = nn.Parameter(torch.Tensor(out_channels_quat))
        self.axis_y = nn.Parameter(torch.Tensor(out_channels_quat))
        self.axis_z = nn.Parameter(torch.Tensor(out_channels_quat))
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize magnitudes
        nn.init.kaiming_uniform_(self.conv_magnitude.weight, a=math.sqrt(5))
        
        # Initialize angles
        nn.init.uniform_(self.conv_theta.weight, -math.pi, math.pi)
        
        # Initialize rotation axes to be unit vectors
        nn.init.uniform_(self.axis_x, -1, 1)
        nn.init.uniform_(self.axis_y, -1, 1)
        nn.init.uniform_(self.axis_z, -1, 1)
        
        # Normalize axes
        with torch.no_grad():
            norm = torch.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
            self.axis_x.div_(norm)
            self.axis_y.div_(norm)
            self.axis_z.div_(norm)

    def forward(self, x):
        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)
            
        batch_size = x.size(0)
        
        # Get magnitude and rotation angle
        magnitudes = F.relu(self.conv_magnitude(x))  # Ensure positive magnitudes
        thetas = self.conv_theta(x)
        
        # Create rotation quaternions
        cos_half_theta = torch.cos(thetas / 2)
        sin_half_theta = torch.sin(thetas / 2)
        
        # Construct rotation quaternions using learned axes
        quat_r = cos_half_theta
        quat_i = sin_half_theta * self.axis_x.view(1, -1, 1, 1)
        quat_j = sin_half_theta * self.axis_y.view(1, -1, 1, 1)
        quat_k = sin_half_theta * self.axis_z.view(1, -1, 1, 1)
        
        # Apply double-sided transformation
        # q' = w * q * conj(w) / ||w||
        left_transform = self._quaternion_multiply(
            [quat_r, quat_i, quat_j, quat_k],
            [x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]]
        )
        
        # Compute conjugate of rotation quaternion
        quat_conj = [quat_r, -quat_i, -quat_j, -quat_k]
        
        # Apply right multiplication by conjugate
        result = self._quaternion_multiply(
            left_transform,
            quat_conj
        )
        
        # Scale by magnitude
        result = [comp * magnitudes for comp in result]
        
        # Stack components
        return torch.stack(result, dim=2)

    def _quaternion_multiply(self, q1, q2):
        """Helper function for quaternion multiplication"""
        r1, i1, j1, k1 = q1
        r2, i2, j2, k2 = q2
        
        r_out = r1*r2 - i1*i2 - j1*j2 - k1*k2
        i_out = r1*i2 + i1*r2 + j1*k2 - k1*j2
        j_out = r1*j2 - i1*k2 + j1*r2 + k1*i2
        k_out = r1*k2 + i1*j2 - j1*i2 + k1*r2
        
        return [r_out, i_out, j_out, k_out]


# Equivariant Quaternion Conv
class QConvEquivariant(nn.Module):
    """
    Equivariant Quaternion Convolution implementation following 
    Shen et al. [151] approach using real-valued kernels.
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
        super(QConvEquivariant, self).__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        # Handle first layer vs subsequent layers - STILL use parameter reduction
        self.is_first_layer = (in_channels == 3)  # RGB input
        if self.is_first_layer:
            actual_in_channels = 1
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
            
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        out_channels_quat = out_channels // 4
            
        # Single real-valued convolution with reduced parameters
        self.conv = Conv(actual_in_channels, out_channels_quat, kernel_size,
                      stride, padding, dilation, groups, bias, 
                      padding_mode, device=self.device, dtype=dtype)
        
        self._initialize_weights()

    def forward(self, x):
        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)
            
        # Split quaternion components
        x_r = x[:, :, 0]  # Real part
        x_i = x[:, :, 1]  # i component
        x_j = x[:, :, 2]  # j component  
        x_k = x[:, :, 3]  # k component

        # Apply real-valued convolution to each component
        # Note: Using same convolution, maintaining equivariance
        y_r = self.conv(x_r)
        y_i = self.conv(x_i) 
        y_j = self.conv(x_j)
        y_k = self.conv(x_k)

        # Stack back into quaternion format [B, C/4, 4, H, W]
        out = torch.stack([y_r, y_i, y_j, y_k], dim=2)
        
        return out



# TRUE QUAT w/double-sided
class NonfuncQconv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super(QConv, self).__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Handle the special case of first layer (RGB input)
        self.is_first_layer = (in_channels == 3)
        
        # For quaternion convolution, we work with quarter of the channels
        self.quat_in_channels = 1 if self.is_first_layer else in_channels // 4
        self.quat_out_channels = out_channels // 4
        
        # Initialize quaternion weights [out_channels/4, in_channels/4, 4, kernel_size, kernel_size]
        self.weight = nn.Parameter(
            torch.randn(self.quat_out_channels, self.quat_in_channels, 4, *self.kernel_size)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.quat_out_channels, 4))
        else:
            self.register_parameter('bias', None)
            
        self._initialize_weights()

    def _initialize_weights(self):
        # Quaternion-aware initialization
        fan_in = self.quat_in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.quat_out_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        
        # Initialize each component
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound_bias = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)

    def rgb_to_quaternion(self, x):
        """Convert RGB input to quaternion format"""
        B, C, H, W = x.shape
        assert C == 3, "Expected RGB input (3 channels)"
        
        # Create quaternion representation [B, 1, 4, H, W]
        # Real part (r) as luminance, and RGB as i,j,k components
        rgb_normalized = (x - x.min()) / (x.max() - x.min())
        return torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
                                          rgb_normalized[:, 0:1], 
                                          rgb_normalized[:, 1:2], 
                                          rgb_normalized[:, 2:3]], dim=1)

    def quaternion_conv2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Implements double-sided quaternion convolution using Hamilton product rules: q * x * q*
        x: Input tensor [batch, in_channels/4, 4, height, width]
        weight: Quaternion kernel [out_channels/4, in_channels/4, 4, kernel_h, kernel_w]
        """
        batch_size = x.size(0)
        in_channels = x.size(1)
        height = x.size(3)
        width = x.size(4)

        # Unfold input for convolution
        x_unf = F.unfold(x.view(batch_size, -1, height, width),
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=self.dilation)

        # Calculate output dimensions
        out_height = ((height + 2 * self.padding[0] - self.dilation[0] * 
                    (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)
        out_width = ((width + 2 * self.padding[1] - self.dilation[1] * 
                    (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)

        kernel_size = self.kernel_size[0] * self.kernel_size[1]
        
        # Reshape unfolded input and weight for quaternion multiplication
        x_unf = x_unf.view(batch_size, in_channels, 4, kernel_size, -1)  # [B, C_in/4, 4, k*k, H*W]
        w = weight.view(self.quat_out_channels, in_channels, 4, -1)      # [C_out/4, C_in/4, 4, k*k]

        # Extract quaternion components
        xr, xi, xj, xk = x_unf.unbind(dim=2)  # Each [B, C_in/4, k*k, H*W]
        wr, wi, wj, wk = w.unbind(dim=2)      # Each [C_out/4, C_in/4, k*k]

        # First Hamilton product (q * x)
        # Sum over input channels and kernel positions
        left_r = (torch.einsum('biks,oik->bos', xr, wr) - 
                torch.einsum('biks,oik->bos', xi, wi) - 
                torch.einsum('biks,oik->bos', xj, wj) - 
                torch.einsum('biks,oik->bos', xk, wk))
        
        left_i = (torch.einsum('biks,oik->bos', xr, wi) + 
                torch.einsum('biks,oik->bos', xi, wr) + 
                torch.einsum('biks,oik->bos', xj, wk) - 
                torch.einsum('biks,oik->bos', xk, wj))
        
        left_j = (torch.einsum('biks,oik->bos', xr, wj) - 
                torch.einsum('biks,oik->bos', xi, wk) + 
                torch.einsum('biks,oik->bos', xj, wr) + 
                torch.einsum('biks,oik->bos', xk, wi))
        
        left_k = (torch.einsum('biks,oik->bos', xr, wk) + 
                torch.einsum('biks,oik->bos', xi, wj) - 
                torch.einsum('biks,oik->bos', xj, wi) + 
                torch.einsum('biks,oik->bos', xk, wr))

        # For the second Hamilton product, we need to handle the dimensions carefully
        # First, prepare the conjugate of weights by summing over input channels and kernel positions
        w_sum_r = torch.sum(wr, dim=(1, 2))  # [C_out/4]
        w_sum_i = -torch.sum(wi, dim=(1, 2))  # Negative for conjugate
        w_sum_j = -torch.sum(wj, dim=(1, 2))  # Negative for conjugate
        w_sum_k = -torch.sum(wk, dim=(1, 2))  # Negative for conjugate

        # Reshape weight sums for broadcasting
        w_sum_r = w_sum_r.view(1, -1, 1)  # [1, C_out/4, 1]
        w_sum_i = w_sum_i.view(1, -1, 1)  # [1, C_out/4, 1]
        w_sum_j = w_sum_j.view(1, -1, 1)  # [1, C_out/4, 1]
        w_sum_k = w_sum_k.view(1, -1, 1)  # [1, C_out/4, 1]

        # Second Hamilton product ((q * x) * q*)
        out_r = (left_r * w_sum_r - left_i * w_sum_i - left_j * w_sum_j - left_k * w_sum_k)
        out_i = (left_r * w_sum_i + left_i * w_sum_r + left_j * w_sum_k - left_k * w_sum_j)
        out_j = (left_r * w_sum_j - left_i * w_sum_k + left_j * w_sum_r + left_k * w_sum_i)
        out_k = (left_r * w_sum_k + left_i * w_sum_j - left_j * w_sum_i + left_k * w_sum_r)

        # Take average
        out_r = 0.5 * out_r
        out_i = 0.5 * out_i
        out_j = 0.5 * out_j
        out_k = 0.5 * out_k

        # Stack quaternion components [B, C_out/4, 4, H*W]
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        # Reshape to final output format [B, C_out/4, 4, H, W]
        out = out.view(batch_size, self.quat_out_channels, 4, out_height, out_width)
        
        return out
    def forward(self, x):
        if self.is_first_layer:
            x = self.rgb_to_quaternion(x)
        
        out = self.quaternion_conv2d(x, self.weight)
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 4, 1, 1)
        
        return out

# TRUE QUAT REGULAR: EXPERIMENTING
class TrueQConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super(TrueQConv, self).__init__()
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        
        # Handle the special case of first layer (RGB input)
        self.is_first_layer = (in_channels == 3)
        
        # For quaternion convolution, we work with quarter of the channels
        self.quat_in_channels = 1 if self.is_first_layer else in_channels // 4
        self.quat_out_channels = out_channels // 4
        
        # Initialize quaternion weights
        kernel_shape = (self.quat_out_channels, self.quat_in_channels, 4, *self.kernel_size)
        self.weight = nn.Parameter(torch.randn(kernel_shape))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.quat_out_channels, 4))
        else:
            self.register_parameter('bias', None)
            
        self._initialize_weights()

    def _initialize_weights(self):
        fan_in = self.quat_in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.quat_out_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            bound_bias = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound_bias, bound_bias)

    def rgb_to_quaternion(self, x):
        """Convert RGB input to quaternion format"""
        B, C, H, W = x.shape
        assert C == 3, "Expected RGB input (3 channels)"
        
        # Create quaternion representation [B, 1, 4, H, W]
        # r = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        # i = x[:, 0:1]  # R channel
        # j = x[:, 1:2]  # G channel
        # k = x[:, 2:3]  # B channel
        # quat = torch.cat([r, i, j, k], dim=1)  # [B, 4, H, W]
        rgb_normalized = (x - x.min()) / (x.max() - x.min())


        quat = torch.cat([rgb_normalized.mean(dim=1, keepdim=True),
                                                rgb_normalized[:, 0:1],
                                                rgb_normalized[:, 1:2],
                                                rgb_normalized[:, 2:3]], dim=1)
        quat = torch.cat([r, i, j, k], dim=1)  # [B, 4, H, W]


        return quat.unsqueeze(1)  # [B, 1, 4, H, W]

    def quaternion_conv2d(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Implements true quaternion convolution using Hamilton product rules.
        x: Input tensor [batch, in_channels/4, 4, height, width]
        weight: Quaternion kernel [out_channels/4, in_channels/4, 4, kernel_h, kernel_w]
        """
        batch_size = x.size(0)
        in_channels = x.size(1)
        height = x.size(3)
        width = x.size(4)

        # Unfold input for convolution
        # [batch, in_c/4 * 4, height, width] -> [batch, in_c/4 * 4, kh*kw, out_h * out_w]
        x_unf = F.unfold(x.view(batch_size, -1, height, width),
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        stride=self.stride,
                        dilation=self.dilation)

        # Calculate output dimensions
        out_height = ((height + 2 * self.padding[0] - self.dilation[0] * 
                    (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1)
        out_width = ((width + 2 * self.padding[1] - self.dilation[1] * 
                    (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1)

        kernel_size = self.kernel_size[0] * self.kernel_size[1]
        
        # Reshape unfolded input and weight for quaternion multiplication
        x_unf = x_unf.view(batch_size, in_channels, 4, kernel_size, -1)  # [B, C_in/4, 4, k*k, H*W]
        w = weight.view(self.quat_out_channels, in_channels, 4, -1)      # [C_out/4, C_in/4, 4, k*k]

        # Extract quaternion components
        xr, xi, xj, xk = x_unf.unbind(dim=2)  # Each [B, C_in/4, k*k, H*W]
        wr, wi, wj, wk = w.unbind(dim=2)      # Each [C_out/4, C_in/4, k*k]

        # Hamilton product using einsum
        # Sum over input channels and kernel positions
        # 'biks,oik->bos' means:
        # b: batch, i: input channels, k: kernel positions, s: spatial positions (H*W)
        # o: output channels
        out_r = (torch.einsum('biks,oik->bos', xr, wr) - 
                torch.einsum('biks,oik->bos', xi, wi) - 
                torch.einsum('biks,oik->bos', xj, wj) - 
                torch.einsum('biks,oik->bos', xk, wk))
        
        out_i = (torch.einsum('biks,oik->bos', xr, wi) + 
                torch.einsum('biks,oik->bos', xi, wr) + 
                torch.einsum('biks,oik->bos', xj, wk) - 
                torch.einsum('biks,oik->bos', xk, wj))
        
        out_j = (torch.einsum('biks,oik->bos', xr, wj) - 
                torch.einsum('biks,oik->bos', xi, wk) + 
                torch.einsum('biks,oik->bos', xj, wr) + 
                torch.einsum('biks,oik->bos', xk, wi))
        
        out_k = (torch.einsum('biks,oik->bos', xr, wk) + 
                torch.einsum('biks,oik->bos', xi, wj) - 
                torch.einsum('biks,oik->bos', xj, wi) + 
                torch.einsum('biks,oik->bos', xk, wr))

        # Stack quaternion components [B, C_out/4, 4, H*W]
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        # Reshape to final output format [B, C_out/4, 4, H, W]
        out = out.view(batch_size, self.quat_out_channels, 4, out_height, out_width)
        
        return out

    def forward(self, x):
        if self.is_first_layer:
            x = self.rgb_to_quaternion(x)
            
        out = self.quaternion_conv2d(x, self.weight)
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 4, 1, 1)
        
        return out

# DOUBLE SIDED CONV WITH MAPPING
class doubleQConv(nn.Module):
    """
    Quaternion Convolution class with double-sided convolution (conjugate).
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

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"

        valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized', 'hamilton', 'poincare']
        assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"

        self.mapping_type = mapping_type
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

        # CONJUGATE PART
        r_conv_conj = r_conv  # Real part stays the same

        i_conv_conj = -i_conv # Imaginary part is negated
        j_conv_conj = -j_conv
        k_conv_conj = -k_conv

        out_r_conj = r_conv_conj
        out_r_conj.sub_(i_conv_conj)
        out_r_conj.sub_(j_conv_conj)
        out_r_conj.sub_(k_conv_conj)

        out_i_conj = r_conv_conj.clone()
        out_i_conj.add_(i_conv_conj)
        out_i_conj.add_(j_conv_conj)
        out_i_conj.sub_(k_conv_conj)

        out_j_conj = r_conv_conj.clone()
        out_j_conj.sub_(i_conv_conj)
        out_j_conj.add_(j_conv_conj)
        out_j_conj.add_(k_conv_conj)

        out_k_conj = r_conv_conj.clone()
        out_k_conj.add_(i_conv_conj)
        out_k_conj.sub_(j_conv_conj)
        out_k_conj.add_(k_conv_conj)


        # COMBINING THE OUTPUT
        out_r_combined = 0.5 * (out_r + out_r_conj)
        out_i_combined = 0.5 * (out_i + out_i_conj)
        out_j_combined = 0.5 * (out_j + out_j_conj)
        out_k_combined = 0.5 * (out_k + out_k_conj)

        # Stack outputs
        out = torch.stack([out_r_combined, out_i_combined, out_j_combined, out_k_combined], dim=2)


        # Clean up intermediate tensors
        del r_conv, i_conv, j_conv, k_conv, r_conv_conj, i_conv_conj, j_conv_conj, k_conv_conj
        del out_r, out_i, out_j, out_k, out_r_conj, out_i_conj, out_j_conj, out_k_conj
        del out_r_combined, out_i_combined, out_j_combined, out_k_combined
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

# DOUBLE SIDED CONV WITHOUT MAPPING

class QConv(nn.Module):
    """
    Quaternion Convolution class implementing double-sided convolution with conjugate
    as described in 'Constructing Convolutional Neural Networks Based on Quaternion'
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
        
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank

        # Define the underlying real-valued convolution for each quaternion component
        Conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank]

        self.is_first_layer = (in_channels == 3)  # For RGB input
        if self.is_first_layer:
            actual_in_channels = 1
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"

        out_channels_quat = out_channels // 4

        # Create convolution layers for each component
        self.conv_r = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias,
                          padding_mode, device=self.device, dtype=dtype)
        
        self.conv_i = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False,
                          padding_mode, device=self.device, dtype=dtype)
        
        self.conv_j = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False,
                          padding_mode, device=self.device, dtype=dtype)
        
        self.conv_k = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False,
                          padding_mode, device=self.device, dtype=dtype)

        self._initialize_weights()

    def _initialize_weights(self):
        # Calculate fan_in
        kernel_prod = np.prod(self.kernel_size)
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod

        # Initialize weights according to the paper's scheme
        nn.init.kaiming_uniform_(self.conv_r.weight, a=math.sqrt(5))
        if self.conv_r.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv_r.bias, -bound, bound)

        # Initialize imaginary components with similar scale
        for conv in [self.conv_i, self.conv_j, self.conv_k]:
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)

        if self.is_first_layer:
            B, Q, H, W = x.shape
            assert Q == 4, "First layer input must have 4 quaternion components"
            
            # Process components
            r_conv = self.conv_r(x[:, 0:1])
            i_conv = self.conv_i(x[:, 1:2])
            j_conv = self.conv_j(x[:, 2:3])
            k_conv = self.conv_k(x[:, 3:4])
        else:
            # For subsequent layers
            x_r = x[:, :, 0, :, :]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]
            with torch.set_grad_enabled(self.training):
                r_conv = self.conv_r(x_r)
                i_conv = self.conv_i(x_i)
                j_conv = self.conv_j(x_j)
                k_conv = self.conv_k(x_k)

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

        # CONJUGATE PART
        r_conv_conj = r_conv  # Real part stays the same

        i_conv_conj = -i_conv # Imaginary part is negated
        j_conv_conj = -j_conv
        k_conv_conj = -k_conv

        out_r_conj = r_conv_conj
        out_r_conj.sub_(i_conv_conj)
        out_r_conj.sub_(j_conv_conj)
        out_r_conj.sub_(k_conv_conj)

        out_i_conj = r_conv_conj.clone()
        out_i_conj.add_(i_conv_conj)
        out_i_conj.add_(j_conv_conj)
        out_i_conj.sub_(k_conv_conj)

        out_j_conj = r_conv_conj.clone()
        out_j_conj.sub_(i_conv_conj)
        out_j_conj.add_(j_conv_conj)
        out_j_conj.add_(k_conv_conj)

        out_k_conj = r_conv_conj.clone()
        out_k_conj.add_(i_conv_conj)
        out_k_conj.sub_(j_conv_conj)
        out_k_conj.add_(k_conv_conj)


        # COMBINING THE OUTPUT
        out_r_combined = 0.5 * (out_r + out_r_conj)
        out_i_combined = 0.5 * (out_i + out_i_conj)
        out_j_combined = 0.5 * (out_j + out_j_conj)
        out_k_combined = 0.5 * (out_k + out_k_conj)

        # Stack outputs
        out = torch.stack([out_r_combined, out_i_combined, out_j_combined, out_k_combined], dim=2)


        # Clean up intermediate tensors
        del r_conv, i_conv, j_conv, k_conv, r_conv_conj, i_conv_conj, j_conv_conj, k_conv_conj
        del out_r, out_i, out_j, out_k, out_r_conj, out_i_conj, out_j_conj, out_k_conj
        del out_r_combined, out_i_combined, out_j_combined, out_k_combined
        return out

    def rgb_to_quaternion(self, rgb_input):
        """Convert RGB to quaternion representation using basic normalization"""
        B, C, H, W = rgb_input.shape
        
        # Normalize RGB values to [0,1]
        rgb_normalized = (rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())
        
        # Create quaternion representation with real part as average intensity
        quat = torch.cat([
            rgb_normalized.mean(dim=1, keepdim=True),  # real part
            rgb_normalized[:, 0:1],  # R -> i
            rgb_normalized[:, 1:2],  # G -> j
            rgb_normalized[:, 2:3]   # B -> k
        ], dim=1)
        
        return quat


class AnalogousQConv(nn.Module):
    """
    Base Quaternion Convolution class.
    """
    def __init__(self, 
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
        rank = 2
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
        
        # Handle first layer vs. subsequent layers
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Initialize real convolution (conv_rr)
        nn.init.kaiming_uniform_(self.conv_rr.weight, a=math.sqrt(5))
        if self.conv_rr.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv_rr.bias, -bound, bound)
        
        # Initialize imaginary parts with smaller weights
        scale_factors = {
            'luminance': [1.0, 0.5, 0.5, 0.5],      # Emphasize luminance
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],  # Slightly more balanced
            'raw_normalized': [1.0, 1.0, 1.0, 1.0]  # Equal emphasis
        }
        
        scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
        convs = [self.conv_ri, self.conv_rj, self.conv_rk]
        for i, conv in enumerate(convs):
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scales[i+1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward pass implementation."""
        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)
        if self.is_first_layer:  # First layer
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
