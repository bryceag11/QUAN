# block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1D, QConv2D, QConv3D
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
from typing import List
from cifar10 import QuaternionMaxPool


def autopad(k: int, p: Optional[int] = None) -> int:
    """Automatic padding calculation based on kernel size.
    
    Args:
        k (int): Kernel size.
        p (int, optional): Desired padding. If None, calculates 'same' padding.
        
    Returns:
        int: Padding size.
    """
    if p is None:
        return (k - 1) // 2
    return p

class QuaternionPolarPool(nn.Module):
    """
    Novel pooling layer that operates in quaternion polar form to preserve 
    rotational relationships while reducing spatial dimensions.
    """
    def __init__(self, kernel_size: int, stride: int = None):
        super(QuaternionPolarPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, 4, H, W]
        B, C, Q, H, W = x.shape
        assert Q == 4, "Quaternion dimension must be 4."
        
        # Reshape to (B, C, H, W)
        x_flat = x.view(B, C, H, W)
        
        # Compute magnitudes and phases for each quaternion
        # Assuming quaternions are normalized; if not, adjust accordingly
        magnitudes = torch.norm(x_flat, dim=1, keepdim=True)  # [B, 1, H, W]
        phases = torch.atan2(x_flat[:, 1:, :, :], x_flat[:, :1, :, :])  # [B, 3, H, W]
        
        # Pool magnitudes using max pooling
        pooled_magnitudes = F.max_pool2d(
            magnitudes, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2
        )  # [B, 1, H', W']
        
        # Pool phases using circular mean
        # Unwrap phases for proper averaging
        cos_phases = torch.cos(phases)
        sin_phases = torch.sin(phases)
        
        pooled_cos = F.avg_pool2d(cos_phases, self.kernel_size, self.stride, padding=self.kernel_size // 2)
        pooled_sin = F.avg_pool2d(sin_phases, self.kernel_size, self.stride, padding=self.kernel_size // 2)
        pooled_phases = torch.atan2(pooled_sin, pooled_cos)  # [B, 3, H', W']
        
        # Reconstruct quaternion
        pooled_real = pooled_magnitudes * torch.cos(pooled_phases[:, 0:1, :, :])
        pooled_i = pooled_magnitudes * torch.sin(pooled_phases[:, 0:1, :, :])
        pooled_j = pooled_magnitudes * torch.sin(pooled_phases[:, 1:2, :, :])
        pooled_k = pooled_magnitudes * torch.sin(pooled_phases[:, 2:3, :, :])
        
        # Concatenate quaternion components
        pooled = torch.cat([pooled_real, pooled_i, pooled_j, pooled_k], dim=1)  # [B, 4, H', W']
        
        return pooled.view(B, C, Q, pooled.shape[2], pooled.shape[3])  # [B, C, 4, H', W']



class InformationTheoreticQuaternionPool(nn.Module):
    """
    Information-Theoretic Quaternion Pooling (ITQPP) layer.
    Emphasizes interchannel relationships by selecting quaternions that maximize mutual information within pooling regions.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initializes the Information-Theoretic Quaternion Pooling layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window.
            stride (int or tuple, optional): Stride of the pooling window. Defaults to kernel_size.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        """
        super(InformationTheoreticQuaternionPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Forward pass for Information-Theoretic Quaternion Pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, 4, H, W).

        Returns:
            torch.Tensor: Pooled tensor with preserved quaternion structure.
        """
        # Ensure channel dimension is a multiple of 4
        batch_size, channels, quat_dim, H, W = x.shape
        assert quat_dim == 4, "Quaternion dimension must be 4."
        assert channels % 4 == 0, "Number of channels must be a multiple of 4."

        # Reshape to separate quaternion components
        x = x.view(batch_size, channels // 4, 4, H, W)  # Shape: (B, C_q, 4, H, W)

        # Apply adaptive pooling to obtain windows
        x_unfold = F.unfold(x.view(batch_size, -1, H, W), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # x_unfold shape: (B, C_q*4*kernel_size*kernel_size, L)

        # Reshape to (B, C_q, 4, kernel_size*kernel_size, L)
        kernel_area = self.kernel_size * self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0] * self.kernel_size[1]
        x_unfold = x_unfold.view(batch_size, channels // 4, quat_dim, kernel_area, -1)

        # Compute entropy for each quaternion across the window
        # Simplified entropy: -sum(p * log(p)), where p is normalized magnitude
        magnitudes = torch.norm(x_unfold, dim=2)  # Shape: (B, C_q, K, L)
        p = magnitudes / (magnitudes.sum(dim=3, keepdim=True) + 1e-8)  # Shape: (B, C_q, K, L)
        entropy = - (p * torch.log(p + 1e-8)).sum(dim=2)  # Shape: (B, C_q, L)

        # Select the quaternion with the highest entropy within each window
        _, indices = entropy.max(dim=1)  # Shape: (B, L)

        # Gather the selected quaternions
        # Create index tensors
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, indices.size(1))
        channel_indices = indices  # Shape: (B, L)

        # Extract quaternions
        pooled_quaternions = x_unfold[batch_indices, channel_indices, :, :, torch.arange(indices.size(1), device=x.device)]

        # Reshape back to (B, C_q*4, H_out, W_out)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        pooled_quaternions = pooled_quaternions.view(batch_size, -1, H_out, W_out)

        return pooled_quaternions

class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for quaternion neural networks.
    Maintains quaternion structure throughout the pooling pyramid.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4"
        
        # Set up intermediate channels (half of in_channels), ensuring multiple of 4
        c_ = in_channels // 2
        assert c_ % 4 == 0, "Hidden channels must be a multiple of 4"

        # First convolution to reduce channel dimensionality
        self.cv1 = QConv2D(in_channels, c_, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Max pooling with kernel size and stride of 1
        self.m = QuaternionMaxPool(kernel_size=kernel_size, stride=1)
        
        # Final convolution after concatenation to project back to out_channels
        self.cv2 = QConv2D(c_ * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SPPF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, 4, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, 4, H, W)
        """
        # Initial convolution
        y = self.cv1(x)  # Shape: (B, c_, 4, H, W)
        
        # Apply three max pooling layers and concatenate outputs
        pooled_outputs = [y]  # Initialize with first convolution output
        for _ in range(3):
            pooled = self.m(pooled_outputs[-1])  # Apply pooling while maintaining quaternion structure
            pooled_outputs.append(pooled)  # Append pooled output
        
        # Concatenate along channel dimension and apply final convolution
        y = torch.cat(pooled_outputs, dim=1)  # Shape: (B, c_ * 4, 4, H, W)
        y = self.cv2(y)  # Project to out_channels: (B, out_channels, 4, H, W)
        return y



# class QBottleneck(nn.Module):
#     """
#     Quaternion Bottleneck with dimension tracking.
#     """
#     def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
#         super().__init__()
#         assert in_channels % 4 == 0 and out_channels % 4 == 0
        
#         self.shortcut = shortcut and in_channels == out_channels
        
#         # First convolution: [B, C_in, 4, H, W] -> [B, C_out, 4, H, W]
#         self.cv1 = QConv2D(in_channels, out_channels, kernel_size=1)
#         self.bn1 = IQBN(out_channels // 4)  # Independent normalization per component
#         self.act1 = QReLU()
        
#         # Second convolution maintains dimensions
#         # Shape: [B, C_out, 4, H, W] -> [B, C_out, 4, H, W]
#         self.cv2 = QConv2D(out_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = IQBN(out_channels // 4)
#         self.act2 = QReLU()
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x
#         out = self.cv1(x)
#         out = self.bn1(out)
#         out = self.act1(out)
#         out = self.cv2(out)
#         out = self.bn2(out)
#         if self.shortcut:
#             out += identity
#         out = self.act2(out)
#         return out

class QuaternionUpsample(nn.Module):
    """
    Custom upsampling module for quaternion tensors.
    Upsamples only the spatial dimensions (H, W), keeping Q intact.
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super(QuaternionUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, Q, H, W]
        
        Returns:
            torch.Tensor: Upsampled tensor of shape [B, C, Q, H*scale_factor, W*scale_factor]
        """

        B, C, Q, H, W = x.shape
        # Reshape to [B * Q, C, H, W] to apply upsampling on spatial dimensions

        # Permute to [B, Q, C, H, W] and make contiguous
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Reshape to [B * Q, C, H, W] to apply upsampling on spatial dimensions
        x = x.view(B * Q, C, H, W)

        # Apply upsampling
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        # Reshape back to [B, Q, C, H_new, W_new]
        H_new, W_new = x.shape[-2], x.shape[-1]
        x = x.view(B, Q, C, H_new, W_new).permute(0, 2, 1, 3, 4).contiguous()

        return x
# 



class C3k2(nn.Module):
    """
    Enhanced C3k2 with parallel processing paths for quaternion features.
    Maintains cross-stage partial design while preserving quaternion structure.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, c3k: bool = False, 
                 shortcut: bool = True, e: float = 0.5):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        # Hidden channels must be multiple of 4 for quaternion ops
        hidden_channels = int(out_channels * e)
        hidden_channels = (hidden_channels // 4) * 4
        
        # Split path
        self.cv1 = QConv2D(in_channels, hidden_channels, 1)
        
        # Main path
        self.cv2 = QConv2D(in_channels, hidden_channels, 1)
        self.m = nn.Sequential(*[
            C3k(hidden_channels, hidden_channels, shortcut=shortcut) if c3k else 
            QBottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
            for _ in range(n)
        ])
        
        # Combine paths
        self.cv3 = QConv2D(2 * hidden_channels, out_channels, 1)
        
        # Batch norm and activation
        self.bn = IQBN(out_channels // 4)
        self.act = QReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel processing.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        # Split processing
        split_features = self.cv1(x)
        
        # Main path with bottleneck blocks
        main_features = self.cv2(x)
        main_features = self.m(main_features)
        
        # Combine paths along channel dimension
        # Preserve quaternion structure during concatenation
        combined = torch.cat([split_features, main_features], dim=1)
        
        # Final processing
        out = self.cv3(combined)
        out = self.bn(out)
        out = self.act(out)
        
        return out

# class C3k2(nn.Module):
#     """
#     Quaternion C3k2 module that conditionally uses C3k based on the `c3k` flag.
    
#     Args:
#         in_channels (int): Number of input channels (must be a multiple of 4).
#         out_channels (int): Number of output channels (must be a multiple of 4).
#         n (int): Number of bottleneck layers.
#         c3k (bool): If True, use C3k instead of C3k2's standard implementation.
#         shortcut (bool): Whether to include residual connections.
#         e (float): Expansion ratio for bottleneck channels.
#     """
#     def __init__(self, in_channels: int, out_channels: int, n: int = 1, c3k: bool = False, 
#                  shortcut: bool = True, e: float = 0.5):
#         super().__init__()
#         assert in_channels % 4 == 0 and out_channels % 4 == 0
        
#         self.c3k = c3k
#         if self.c3k:
#             self.module = C3k(in_channels, out_channels, n=n, shortcut=shortcut, e=e)
#         else:
#             hidden_channels = int(out_channels * e)
#             hidden_channels = (hidden_channels // 4) * 4  # Ensure multiple of 4
            
#             # Shape: [B, C_in, 4, H, W] -> [B, hidden_ch, 4, H, W]
#             self.cv1 = QConv2D(in_channels, hidden_channels, 1)
#             self.bn1 = IQBN(hidden_channels // 4)  # Normalizes each quaternion component independently
#             self.act1 = QReLU()
            
#             # Shape maintained: [B, hidden_ch, 4, H, W] -> [B, hidden_ch, 4, H, W]
#             self.m = nn.Sequential(*[
#                 QBottleneck(hidden_channels, hidden_channels, shortcut) 
#                 for _ in range(n)
#             ])
            
#             # Shape: [B, hidden_ch, 4, H, W] -> [B, C_out, 4, H, W]
#             self.cv2 = QConv2D(hidden_channels, out_channels, 1)
#             self.bn2 = IQBN(out_channels // 4)
#             self.act2 = QReLU()
#         # Initialize weights at end of init
#         for m in self.modules():
#             if isinstance(m, QConv2D):
#                 nn.init.kaiming_normal_(m.conv_rr.weight, mode='fan_out', nonlinearity='relu')
#                 if m.conv_rr.bias is not None:
#                     nn.init.constant_(m.conv_rr.bias, 0)
                    
#                 for conv in [m.conv_ri, m.conv_rj, m.conv_rk]:
#                     nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
#                     if conv.bias is not None:
#                         nn.init.constant_(conv.bias, 0)
                        
#             elif isinstance(m, IQBN):
#                 if hasattr(m, 'gamma'):
#                     nn.init.constant_(m.gamma, 1.0)
#                 if hasattr(m, 'beta'):
#                     nn.init.constant_(m.beta, 0.0)
                    
#             elif isinstance(m, (QBN, nn.BatchNorm2d)):
#                 if hasattr(m, 'weight') and m.weight is not None:
#                     nn.init.constant_(m.weight, 1.0)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.c3k:
#             return self.module(x)
#         else:
#             x = self.cv1(x)
#             x = self.bn1(x)
#             x = self.act1(x)
#             x = self.m(x)
#             x = self.cv2(x)
#             x = self.bn2(x)
#             return self.act2(x)

class C3k(nn.Module):
    """
    C3k module for quaternion data - CSP bottleneck with customizable kernel sizes.
    This version is designed to work with the parallel C3k2 implementation.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, 
                 e: float = 0.5, g: int = 1, k: int = 3):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiple of 4"
        
        # Calculate hidden channels (ensure multiple of 4)
        hidden_channels = int(out_channels * e)
        hidden_channels = (hidden_channels // 4) * 4
        
        # First conv reduces channels
        self.cv1 = QConv2D(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.bn1 = IQBN(hidden_channels // 4)
        self.act1 = QReLU()
        
        # Bottleneck block sequence
        self.m = nn.Sequential(*[
            QBottleneck(
                hidden_channels, 
                hidden_channels,
                shortcut=shortcut,
                k=k,
                groups=g
            ) for _ in range(n)
        ])
        
        # Final conv restores channels
        self.cv2 = QConv2D(hidden_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = IQBN(out_channels // 4)
        self.act2 = QReLU()
        
        self.shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass maintaining quaternion structure.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        # Store identity if using shortcut
        identity = x if self.shortcut else None
        
        # Initial convolution
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Process through bottlenecks
        x = self.m(x)
        
        # Final convolution
        x = self.cv2(x)
        x = self.bn2(x)
        
        # Add shortcut if enabled
        if self.shortcut:
            x = x + identity
            
        x = self.act2(x)
        
        return x

class QBottleneck(nn.Module):
    """
    Quaternion-aware bottleneck block used in C3k.
    """
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, 
                 groups: int = 1, k: int = 3):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        mid_channels = in_channels // 2  # Reduction in bottleneck
        mid_channels = (mid_channels // 4) * 4  # Ensure multiple of 4
        
        # First 1x1 conv to reduce channels
        self.cv1 = QConv2D(in_channels, mid_channels, kernel_size=1)
        self.bn1 = IQBN(mid_channels)
        self.act1 = QReLU()
        
        # kxk conv
        self.cv2 = QConv2D(
            mid_channels, 
            mid_channels, 
            kernel_size=k,
            stride=1,
            padding=k//2
        )
        self.bn2 = IQBN(mid_channels)
        self.act2 = QReLU()
        
        # 1x1 conv to restore channels
        self.cv3 = QConv2D(mid_channels, out_channels, kernel_size=1)
        self.bn3 = IQBN(out_channels)
        self.act3 = QReLU()
        
        self.shortcut = shortcut and in_channels == out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quaternion bottleneck.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        identity = x if self.shortcut else None
        
        # First conv + bn + relu
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Second conv + bn + relu
        x = self.cv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Third conv + bn
        x = self.cv3(x)
        x = self.bn3(x)
        
        # Add shortcut if enabled
        if self.shortcut:
            x = x + identity
            
        x = self.act3(x)
        
        return x


class Reshape(nn.Module):
    """
    Reshape Layer.
    
    Args:
        shape (list or tuple): Desired output shape. Use -1 to infer dimensions.
    """
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape  # Example: [B, 4, -1, H, W]

    def forward(self, x):
        return x.view(*self.shape)

class QAttention(nn.Module):
    """
    Quaternion Attention module performing self-attention on quaternion-structured input tensors.
    Properly handles 5D input tensors (batch, channels, quaternion_dim, height, width).
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 1.0):
        super(QAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # Total output channels for qkv (query, key, value)
        nh_kd = self.num_heads * self.key_dim
        h = nh_kd * 3  # For q, k, v
        
        # Quaternion-aware convolutions
        self.qkv = QConv2D(dim, h, kernel_size=1, stride=1)
        self.proj = QConv2D(dim, dim, kernel_size=1, stride=1)
        self.pe = QConv2D(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        
        # Initialize weights
        for layer in [self.qkv, self.proj, self.pe]:
            if isinstance(layer, QConv2D):
                nn.init.kaiming_uniform_(layer.conv_rr.weight, a=math.sqrt(5))
                if layer.conv_rr.bias is not None:
                    nn.init.constant_(layer.conv_rr.bias, 0)
                nn.init.kaiming_uniform_(layer.conv_ri.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(layer.conv_rj.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(layer.conv_rk.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Quaternion Attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W)
        
        Returns:
            torch.Tensor: Output tensor after self-attention, shape (B, C, 4, H, W)
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Quaternion dimension must be 4."
        assert C % self.num_heads == 0, "Channels must be divisible by num_heads."
        
        # Reshape to (B, C, H, W)
        x_flat = x.view(B, C, H, W)
        
        # Compute Q, K, V
        qkv = self.qkv(x_flat)  # [B, 3*nh_kd, H, W]
        qkv = qkv.chunk(3, dim=1)  # [B, nh_kd, H, W] each
        q, k, v = qkv
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.key_dim, H * W)  # [B, num_heads, key_dim, H*W]
        k = k.view(B, self.num_heads, self.key_dim, H * W)  # [B, num_heads, key_dim, H*W]
        v = v.view(B, self.num_heads, self.head_dim, H * W)  # [B, num_heads, head_dim, H*W]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, key_dim, key_dim]
        attn = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, key_dim, key_dim]
        
        # Apply attention to V
        out = torch.matmul(attn, v)  # [B, num_heads, key_dim, head_dim]
        
        # Reshape and project
        out = out.view(B, self.num_heads * self.key_dim, H, W)  # [B, num_heads * key_dim, H, W]
        out = self.proj(out)  # [B, dim, H, W]
        
        # Add positional embedding
        out = out + self.pe(x_flat)  # [B, dim, H, W]
        
        # Reshape back to quaternion structure
        out = out.view(B, C, 4, H, W)  # [B, C, 4, H, W]
        
        return out



class PSABlock(nn.Module):
    """
    Position Sensitive Attention Block with quaternion support.
    """
    def __init__(self, c: int, attn_ratio: float = 1.0, num_heads: int = 8, shortcut: bool = True):
        super().__init__()
        assert c % 4 == 0, "Channels must be multiple of 4"
        self.Q = 4  # Quaternion dimension

        # Attention: [B, C, 4, H, W] -> [B, C, 4, H, W]
        # Splits into Q,K,V while maintaining quaternion structure
        self.attn = QAttention(dim=c, num_heads=num_heads, attn_ratio=attn_ratio)
        
        # FFN with quaternion-aware operations
        # Shape: [B, C, 4, H, W] -> [B, 2C, 4, H, W] -> [B, C, 4, H, W]
        self.ffn = nn.Sequential(
            QConv2D(c, (c * 2), 1),
            IQBN((c * 2)),
            QReLU(),
            QConv2D((c * 2), c , 1),
            IQBN(c )
        )
        self.shortcut = shortcut

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W] where C is already quaternion-adjusted
        """
        x = x + self.attn(x) if self.shortcut else self.attn(x)
        x = x + self.ffn(x) if self.shortcut else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in quaternion neural networks.
    
    Combines quaternion-specific convolutions with attention mechanisms for enhanced feature extraction.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        e (float): Expansion ratio for hidden channels.
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super(PSA, self).__init__()
        assert c1 % 4 == 0 and c2 % 4 == 0, "Input and output channels must be multiples of 4."
        
        self.c = int(c1 * e)
        self.c = (self.c // 4) * 4  # Ensure hidden channels are multiples of 4
        assert self.c > 0, "Hidden channels must be positive and a multiple of 4."
        
        # Quaternion-aware convolution to reduce channels
        self.cv1 = QConv2D(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # Split into two parts: 'a' and 'b'
        # 'a' will go through attention and FFN
        # 'b' will remain unchanged
        self.attn = QAttention(dim=self.c, num_heads=self.c // (4 * 4), attn_ratio=1.0)
        
        # Feed-Forward Network for 'a'
        self.ffn = nn.Sequential(
            QConv2D(self.c, self.c * 2, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c * 2),
            QReLU(),
            QConv2D(self.c * 2, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c)
        )
        
        # Final convolution to restore channels
        self.cv2 = QConv2D(2 * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        # Initial convolution, normalization, and activation
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Split into 'a' and 'b'
        a, b = x.chunk(2, dim=1)  # Each has channels = self.c
        
        # Apply attention to 'a'
        a = self.attn(a)
        a = self.ffn(a)
        
        # Concatenate 'a' and 'b'
        out = torch.cat((a, b), dim=1)  # Shape: (B, 2 * self.c, H, W)
        
        # Final convolution, normalization, and activation
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        return out

class QAttention(nn.Module):
    """
    Quaternion-aware attention module that performs self-attention while preserving quaternion structure.
    
    Args:
        dim (int): The input quaternion channels (already divided by 4)
        num_heads (int): Number of attention heads
        attn_ratio (float): Ratio to determine key dimension
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # Calculate dimensions for Q, K, V
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2  # Total hidden dim
        
        # Quaternion-aware layers
        self.qkv = QConv2D(dim, h, kernel_size=1)  # For Q, K, V projection
        self.proj = QConv2D(dim, dim, kernel_size=1)  # Output projection
        self.pe = QConv2D(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4)  # Positional encoding
        
        # Layer normalization for stability
        self.norm = IQBN(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass preserving quaternion structure.
        
        Args:
            x: Input tensor [B, C, 4, H, W] where C is quaternion channels
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input (Q=4)"
        N = H * W
        
        # Project to Q, K, V while preserving quaternion structure
        qkv = self.qkv(x)  # [B, h, 4, H, W]
        
        # Reshape and split maintaining quaternion components
        qkv = qkv.reshape(B, -1, Q, N)  # [B, h, 4, HW]
        chunks = [self.key_dim, self.key_dim, self.head_dim]
        q, k, v = qkv.view(B, self.num_heads, -1, Q, N).split(chunks, dim=2)
        
        # Compute attention scores
        # Handle each quaternion component separately while maintaining relationships
        q = q.permute(0, 1, 3, 2, 4)  # [B, num_heads, 4, key_dim, HW]
        k = k.permute(0, 1, 3, 4, 2)  # [B, num_heads, 4, HW, key_dim]
        v = v.permute(0, 1, 3, 4, 2)  # [B, num_heads, 4, HW, head_dim]
        
        # Compute attention with quaternion structure preservation
        attn = torch.matmul(q, k) * self.scale  # [B, num_heads, 4, key_dim, key_dim]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attn, v)  # [B, num_heads, 4, key_dim, head_dim]
        x = x.permute(0, 1, 3, 2, 4).reshape(B, C, Q, H, W)
        
        # Add positional encoding to enhance spatial information
        pe = self.pe(x)
        x = x + pe
        
        # Final projection and normalization
        x = self.proj(x)
        x = self.norm(x)
        
        return x

class MaxSigmoidAttnBlock(nn.Module):
    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        super().__init__()
        # Convert input channels to quaternion channels
        quat_c1 = c1 // 4
        quat_c2 = c2 // 4
        quat_ec = ec // 4
        
        self.nh = nh
        self.hc = quat_c2 // nh  # Already in quaternion channels
        
        # Quaternion embedding conv if needed
        self.ec = QConv2D(quat_c1, quat_ec, kernel_size=1) if quat_c1 != quat_ec else None
        
        # Guide linear layer remains unchanged as it processes regular features
        self.gl = nn.Linear(gc, quat_ec * 4)  # Multiply by 4 to match quaternion channels
        
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = QConv2D(quat_c1, quat_c2, kernel_size=3, stride=1, padding=1)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
    
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input"
        
        # Process guide
        guide = self.gl(guide)  # [B, ec]
        guide = guide.view(B, -1, self.nh, self.hc)
        
        # Process input
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(B, Q, self.nh, self.hc, H, W)
        
        # Compute attention with quaternion structure preservation
        aw = torch.einsum('bqnchw,bnmc->bmhwn', embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale
        
        # Apply attention
        x = self.proj_conv(x)  # [B, C2, 4, H, W]
        x = x.view(B, self.nh, -1, Q, H, W)
        x = x * aw.unsqueeze(2).unsqueeze(2)
        
        return x.view(B, -1, Q, H, W)

class C2PSA(nn.Module):
    """C2PSA module with proper quaternion handling."""
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5):
        super(C2PSA, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4 for quaternions."
        
        quat_c1 = in_channels 
        quat_c2 = out_channels
        
        # Calculate hidden channels ensuring multiple of 4
        self.c = int(quat_c2 * e)  # e.g., if c2=1024, quat_c2=256, e=0.5 -> c=128
        
        # Quaternion-aware convolution to reduce channels
        self.cv1 = QConv2D(quat_c1, 2 * self.c, kernel_size=1)  # e.g., 256 -> 256
        self.cv2 = QConv2D(2 * self.c, quat_c2, kernel_size=1)  # e.g., 256 -> 256
        
        # PSA blocks
        self.m = nn.Sequential(*[
            PSABlock(
                c=self.c,  # This is already in quaternion channels
                attn_ratio=0.5,
                num_heads=max(1, self.c // 64)  # Adjust head count appropriately
            ) for _ in range(n)
        ])
        
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C//4, 4, H, W] 
               e.g., for c1=1024: [B, 256, 4, H, W]
        """
        # Split while preserving quaternion structure
        features = self.cv1(x)  # [B, 2*c, 4, H, W]
        a, b = features.chunk(2, dim=1)  # Each [B, c, 4, H, W]
        
        # Process through attention blocks
        b = self.m(b)
        
        # Combine and project
        out = self.cv2(torch.cat([a, b], dim=1))  # Back to [B, C//4, 4, H, W]
        return out

   
class QuaternionPyramidAttention(nn.Module):
    """
    Novel block: Multi-scale quaternion attention with rotation invariance
    - Processes features at multiple scales
    - Maintains quaternion structure
    - Computationally efficient
    """
    def __init__(self, channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.qatt_blocks = nn.ModuleList([
            QAttention(channels//4) for _ in scales
        ])
        
    def forward(self, x):
        results = []
        for scale, qatt in zip(self.scales, self.qatt_blocks):
            # Pool, attend, upsample while preserving quaternion structure
            pooled = F.avg_pool2d(x, scale)
            attended = qatt(pooled)
            upsampled = F.interpolate(attended, size=x.shape[-2:])
            results.append(upsampled)
        return torch.cat(results, dim=1)

class QuaternionFeatureFusion(nn.Module):
    """
    Novel block: Quaternion-aware feature fusion
    - Dynamically weights feature combinations
    - Preserves rotational equivariance
    """
    def __init__(self, channels):
        super().__init__()
        self.qconv = QConv2D(channels, channels//4, 1)
        self.qatt = QAttention(channels//4)
        
    def forward(self, x1, x2):
        # Fusion while maintaining quaternion properties
        fused = self.qconv(torch.cat([x1, x2], dim=1))
        weighted = self.qatt(fused)
        return weighted
    
class HeadC3k2(nn.Module):
    """
    C3k2 block for detection head that converts quaternion features to real-valued features.
    Maintains cross-stage partial design while handling quaternion-to-real conversion.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, 
                 convert_to_real: bool = True, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.convert_to_real = convert_to_real
        
        # Ensure input channels are multiple of 4 for quaternion ops
        assert in_channels % 4 == 0, "Input channels must be multiple of 4"
        
        # For quaternion processing, out_channels should be multiple of 4
        quat_out_channels = ((out_channels + 3) // 4) * 4
        
        # Hidden channels calculation
        hidden_channels = int(quat_out_channels * e)
        hidden_channels = (hidden_channels // 4) * 4
        
        # Split path
        self.cv1 = QConv2D(in_channels, hidden_channels, 1)
        
        # Main path with bottleneck blocks
        self.cv2 = QConv2D(in_channels, hidden_channels, 1)
        self.m = nn.Sequential(*[
            QBottleneck(hidden_channels, hidden_channels, shortcut)
            for _ in range(n)
        ])
        
        # Quaternion to real conversion layers if needed
        if self.convert_to_real:
            # Convert quaternion features to real using learned weights
            self.quat_to_real = nn.Sequential(
                # Combine quaternion components
                QConv2D(2 * hidden_channels, hidden_channels, 1),
                IQBN(hidden_channels // 4),
                QReLU(),
                
                # Project to real space
                nn.Conv2d(hidden_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            )
        else:
            # Stay in quaternion space
            self.cv3 = QConv2D(2 * hidden_channels, quat_out_channels, 1)
            self.bn = IQBN(quat_out_channels // 4)
            self.act = QReLU()
            
    def _combine_quaternion_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines quaternion components with learned weights.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Combined features [B, C, H, W]
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input with 4 components"
        
        # Split quaternion components
        r, i, j, k = x.unbind(dim=2)
        
        # Combine components with learned weights and nonlinearity
        combined = torch.stack([r, i, j, k], dim=1)  # [B, 4, C, H, W]
        weights = F.softmax(self.quat_weights, dim=0)  # [4]
        combined = (combined * weights.view(1, 4, 1, 1, 1)).sum(dim=1)  # [B, C, H, W]
        
        return combined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quaternion to real conversion.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            If convert_to_real: Output tensor [B, C_out, H, W]
            Else: Output tensor [B, C_out, 4, H, W]
        """
        # Split processing
        split_features = self.cv1(x)
        
        # Main path with bottleneck blocks
        main_features = self.cv2(x)
        main_features = self.m(main_features)
        
        # Combine paths while preserving quaternion structure
        combined = torch.cat([split_features, main_features], dim=1)
        
        if self.convert_to_real:
            # Convert to real-valued features
            out = self.quat_to_real(combined)
        else:
            # Keep as quaternion features
            out = self.cv3(combined)
            out = self.bn(out)
            out = self.act(out)
            
        return out

class QRotationAttention(nn.Module):
    """
    Rotation-aware attention block specifically for OBB detection.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Project features while preserving quaternion structure
        self.q = QConv2D(channels, channels, 1)
        self.k = QConv2D(channels, channels, 1)
        self.v = QConv2D(channels, channels, 1)
        
        # Output projection
        self.proj = QConv2D(channels, channels, 1)
        self.norm = IQBN(channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features with rotation attention [B, C, 4, H, W]
        """
        B, C, Q, H, W = x.shape
        
        # Project to Q,K,V while keeping quaternion structure
        q = self.q(x)  # [B, C, 4, H, W]
        k = self.k(x)  # [B, C, 4, H, W]
        v = self.v(x)  # [B, C, 4, H, W]
        
        # Reshape for attention computation
        q = q.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        k = k.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        v = v.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        
        # Compute rotation-aware attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C//4)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention and reshape
        out = torch.matmul(attn, v)  # [B, 4, C/4, H*W]
        out = out.permute(0, 2, 1, 3).reshape(B, C, 4, H, W)
        
        # Project and normalize
        out = self.proj(out)
        out = self.norm(out)
        
        return out

class QOBBFeatureFusion(nn.Module):
    """
    Feature fusion block specifically designed for OBB detection.
    Preserves and enhances orientation information during feature fusion.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        # Dimension reduction
        self.conv1 = QConv2D(in_channels, out_channels, 1)
        self.norm1 = IQBN(out_channels)
        self.act = QReLU()
        
        # Rotation-specific attention
        self.rot_attn = QRotationAttention(out_channels)
        
        # Channel attention with quaternion structure preservation
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(out_channels, out_channels//4, 1),
            QReLU(),
            QConv2D(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, 4, H, W]
        Returns:
            Fused features with enhanced orientation information [B, C_out, 4, H, W]
        """
        # Initial convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        
        # Enhance rotation features
        out = self.rot_attn(out)
        
        # Apply channel attention while preserving quaternion structure
        w = self.ca(out.view(out.shape[0], -1, *out.shape[-2:]))
        w = w.view_as(out)
        out = out * w
        
        return out

class QRotationCrossAttention(nn.Module):
    """
    Cross-attention module specifically for OBB detection.
    Enhances feature interaction while preserving orientation information.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Projections for cross attention
        self.q_proj = QConv2D(channels, channels, kernel_size=1)
        self.k_proj = QConv2D(channels, channels, kernel_size=1)
        self.v_proj = QConv2D(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_proj = QConv2D(channels, channels, kernel_size=1)
        self.norm = IQBN(channels)
        
        # Quaternion-specific angle attention
        self.angle_attn = nn.Sequential(
            QConv2D(channels, channels//4, 1),
            QReLU(),
            QConv2D(channels//4, 4, 1)  # 4 for quaternion components
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Current level features [B, C, 4, H, W]
            x2: Cross level features [B, C, 4, H, W]
        Returns:
            Enhanced features with rotation-aware cross attention [B, C, 4, H, W]
        """
        B, C, Q, H, W = x1.shape
        
        # Project to Q,K,V
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        
        # Compute quaternion-specific angle attention
        angle_weights = self.angle_attn(x1)  # [B, 4, 4, H, W]
        angle_weights = angle_weights.softmax(dim=1)
        
        # Apply cross attention with angle weighting
        q = q * angle_weights
        k = k * angle_weights
        
        # Reshape and compute attention
        q = q.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        k = k.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        v = v.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C//4)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, C, 4, H, W)
        
        # Project and normalize
        out = self.out_proj(out)
        out = self.norm(out)
        
        return out

class QAdaptiveFeatureExtraction(nn.Module):
    """
    Enhanced feature extraction with multi-scale processing and channel attention.
    Shape: [B, C, 4, H, W] -> [B, C, 4, H, W]
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        self.channels = channels
        mid_channels = max(channels // reduction_ratio, 32)
        
        # Multi-scale branches
        self.local_branch = nn.Sequential(
            QConv2D(channels, channels//2, kernel_size=3, padding=1),
            IQBN(channels//2),
            QReLU()
        )
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//2, kernel_size=1),
            QReLU()
        )
        
        # Channel attention
        self.ca = nn.Sequential(
            QConv2D(channels, mid_channels, 1),
            QReLU(),
            QConv2D(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = QConv2D(channels, channels, 3, padding=1)
        self.norm = IQBN(channels)
        self.act = QReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features [B, C, 4, H, W]
        """
        # Process branches
        local_feat = self.local_branch(x)  # [B, C/2, 4, H, W]
        
        # Global context
        global_feat = self.global_branch(x)  # [B, C/2, 4, 1, 1]
        global_feat = global_feat.expand(-1, -1, -1, x.shape[-2], x.shape[-1])
        
        # Combine features
        combined = torch.cat([local_feat, global_feat], dim=1)  # [B, C, 4, H, W]
        
        # Apply channel attention
        attn = self.ca(combined)
        out = combined * attn
        
        # Final refinement
        out = self.refine(out)
        out = self.norm(out)
        out = self.act(out)
        
        return out


class QDualAttention(nn.Module):
    """
    Dual path attention combining spatial and channel attention
    with quaternion structure preservation.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Spatial attention branch - ensure output is multiple of 4
        self.spatial = nn.Sequential(
            QConv2D(channels, channels//8, 1),
            QReLU(),
            QConv2D(channels//8, 4, 1),  # Changed from 1 to 4 channels
            nn.Sigmoid()
        )
        
        # Channel attention branch
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//8, 1),
            QReLU(),
            QConv2D(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = QConv2D(channels, channels, 3, padding=1)
        self.norm = IQBN(channels)
        self.act = QReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features [B, C, 4, H, W]
        """
        # Spatial attention - repeat attention map for all quaternion components
        spatial_attn = self.spatial(x)
        spatial_attn = spatial_attn.repeat(1, x.size(1)//4, 1, 1)  # Repeat to match input channels
        spatial_out = x * spatial_attn
        
        # Channel attention
        channel_attn = self.channel(x)
        channel_out = x * channel_attn
        
        # Combine and refine
        out = spatial_out + channel_out
        out = self.refine(out)
        out = self.norm(out)
        out = self.act(out)
        
        return out

class QAdaptiveFusion(nn.Module):
    """
    Adaptive feature fusion with dynamic weighting
    and enhanced quaternion feature interaction.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Feature transformation
        self.transform1 = QConv2D(channels, channels//2, 1)
        self.transform2 = QConv2D(channels, channels//2, 1)
        
        # Dynamic weight prediction
        self.weight_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//4, 1),
            QReLU(),
            QConv2D(channels//4, 2, 1),  # 2 weights for 2 paths
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            QConv2D(channels//2, channels//2, 3, padding=1),
            IQBN(channels//2),
            QReLU(),
            QConv2D(channels//2, channels//2, 3, padding=1)
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Input tensors [B, C, 4, H, W]
        Returns:
            Fused features [B, C//2, 4, H, W]
        """
        # Transform features
        f1 = self.transform1(x1)
        f2 = self.transform2(x2)
        
        # Predict fusion weights
        weights = self.weight_pred(torch.cat([f1, f2], dim=1))
        
        # Weighted fusion
        fused = f1 * weights[:, 0:1, :, :] + f2 * weights[:, 1:2, :, :]
        
        # Refine fused features
        out = self.refine(fused)
        
        return out
