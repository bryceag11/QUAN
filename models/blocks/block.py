# block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1D, QConv2D, QConv3D
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
from typing import List



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


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (QSPPF) layer for quaternion neural networks.

    This module applies a series of max pooling operations with the same kernel size
    and concatenates the results along the channel dimension, followed by a convolution
    to mix the features.

    Args:
        in_channels (int): Number of input channels (must be a multiple of 4).
        out_channels (int): Number of output channels (must be a multiple of 4).
        kernel_size (int): Kernel size for max pooling. Defaults to 5.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super(SPPF, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4 for quaternions."
        
        # Set up intermediate channels (half of in_channels), ensuring they are a multiple of 4
        c_ = in_channels // 2
        assert c_ % 4 == 0, "Hidden channels must be a multiple of 4 for quaternions."

        # First convolution to reduce channel dimensionality
        self.cv1 = QConv2D(in_channels, c_, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Max pooling with kernel size and stride of 1
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Final convolution after concatenation to project back to out_channels
        self.cv2 = QConv2D(c_ * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SPPF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, 4, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, 4, H, W).
        """
        # Initial convolution
        y = self.cv1(x)
        y1 = self.m(y)          # Pool once
        y2 = self.m(self.m(y))  # Pool twice 
        y3 = self.m(self.m(self.m(y)))  # Pool three times
        out = torch.cat([y, y1, y2, y3], dim=1)
        return self.cv2(out)



class QBottleneck(nn.Module):
    """
    Quaternion Bottleneck with dimension tracking.
    """
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        self.shortcut = shortcut and in_channels == out_channels
        
        # First convolution: [B, C_in, 4, H, W] -> [B, C_out, 4, H, W]
        self.cv1 = QConv2D(in_channels, out_channels, kernel_size=1)
        self.bn1 = IQBN(out_channels // 4)  # Independent normalization per component
        self.act1 = QReLU()
        
        # Second convolution maintains dimensions
        # Shape: [B, C_out, 4, H, W] -> [B, C_out, 4, H, W]
        self.cv2 = QConv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = IQBN(out_channels // 4)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.cv2(out)
        out = self.bn2(out)
        if self.shortcut:
            out += identity
        out = self.act2(out)
        return out

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
    Quaternion C3k2 module that conditionally uses C3k based on the `c3k` flag.
    
    Args:
        in_channels (int): Number of input channels (must be a multiple of 4).
        out_channels (int): Number of output channels (must be a multiple of 4).
        n (int): Number of bottleneck layers.
        c3k (bool): If True, use C3k instead of C3k2's standard implementation.
        shortcut (bool): Whether to include residual connections.
        e (float): Expansion ratio for bottleneck channels.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, c3k: bool = False, 
                 shortcut: bool = True, e: float = 0.5):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        self.c3k = c3k
        if self.c3k:
            self.module = C3k(in_channels, out_channels, n=n, shortcut=shortcut, e=e)
        else:
            hidden_channels = int(out_channels * e)
            hidden_channels = (hidden_channels // 4) * 4  # Ensure multiple of 4
            
            # Shape: [B, C_in, 4, H, W] -> [B, hidden_ch, 4, H, W]
            self.cv1 = QConv2D(in_channels, hidden_channels, 1)
            self.bn1 = IQBN(hidden_channels // 4)  # Normalizes each quaternion component independently
            self.act1 = QReLU()
            
            # Shape maintained: [B, hidden_ch, 4, H, W] -> [B, hidden_ch, 4, H, W]
            self.m = nn.Sequential(*[
                QBottleneck(hidden_channels, hidden_channels, shortcut) 
                for _ in range(n)
            ])
            
            # Shape: [B, hidden_ch, 4, H, W] -> [B, C_out, 4, H, W]
            self.cv2 = QConv2D(hidden_channels, out_channels, 1)
            self.bn2 = IQBN(out_channels // 4)
            self.act2 = QReLU()
        # Initialize weights at end of init
        for m in self.modules():
            if isinstance(m, QConv2D):
                nn.init.kaiming_normal_(m.conv_rr.weight, mode='fan_out', nonlinearity='relu')
                if m.conv_rr.bias is not None:
                    nn.init.constant_(m.conv_rr.bias, 0)
                    
                for conv in [m.conv_ri, m.conv_rj, m.conv_rk]:
                    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                    if conv.bias is not None:
                        nn.init.constant_(conv.bias, 0)
                        
            elif isinstance(m, IQBN):
                if hasattr(m, 'gamma'):
                    nn.init.constant_(m.gamma, 1.0)
                if hasattr(m, 'beta'):
                    nn.init.constant_(m.beta, 0.0)
                    
            elif isinstance(m, (QBN, nn.BatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c3k:
            return self.module(x)
        else:
            x = self.cv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.m(x)
            x = self.cv2(x)
            x = self.bn2(x)
            return self.act2(x)

class C3k(nn.Module):
    """
    Quaternion C3k module.
    
    Args:
        in_channels (int): Number of input channels (must be a multiple of 4).
        out_channels (int): Number of output channels (must be a multiple of 4).
        n (int): Number of bottleneck layers.
        shortcut (bool): Whether to include residual connections.
        e (float): Expansion ratio for bottleneck channels.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super(C3k, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4 for quaternions."
        hidden_channels = int(out_channels * e)
        hidden_channels = (hidden_channels // 4) * 4  # Ensure multiple of 4
        
        self.m = nn.Sequential(
            *[QBottleneck(in_channels, hidden_channels, shortcut) for _ in range(n)]
        )
        self.cv = QConv2D(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = IQBN(out_channels // 4)
        self.act = QReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.cv(self.m(x))))


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
        self.attn = QAttention(dim=c // self.Q, num_heads=num_heads, attn_ratio=attn_ratio)
        
        # FFN with quaternion-aware operations
        # Shape: [B, C, 4, H, W] -> [B, 2C, 4, H, W] -> [B, C, 4, H, W]
        self.ffn = nn.Sequential(
            QConv2D(c // self.Q, (c * 2) // self.Q, 1),
            IQBN((c * 2) // self.Q),
            QReLU(),
            QConv2D((c * 2) // self.Q, c // self.Q, 1),
            IQBN(c // self.Q)
        )
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PSABlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W]
        """
        # Apply attention
        out = self.attn(x)  # [B, C//Q, H, W]
        if self.shortcut:
            out = out + x  # Residual connection

        # Apply Feed-Forward Network
        out_ffn = self.ffn(out)  # [B, C//Q, H, W]
        if self.shortcut:
            out_ffn = out_ffn + out  # Residual connection

        return out_ffn


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

class MaxSigmoidAttnBlock(nn.Module):
    """Quaternion-aware Max Sigmoid Attention Block."""
    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        super().__init__()
        assert c1 % 4 == 0 and c2 % 4 == 0, "Channels must be multiples of 4"
        
        self.Q = 4  # Quaternion dimension
        self.nh = nh
        self.hc = c2 // nh
        assert self.hc % 4 == 0, "Head channels must be multiple of 4"
        
        # Optional embedding convolution
        if c1 != ec:
            self.ec_conv = QConv2D(c1 // self.Q, ec // self.Q, kernel_size=1, stride=1)
            self.bn_ec = IQBN(ec // self.Q)
            self.act_ec = QReLU()
        else:
            self.ec_conv = None
        
        # Guide processing
        self.gl = nn.Linear(gc, ec)
        
        # Attention parameters
        self.bias = nn.Parameter(torch.zeros(nh))
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else torch.tensor(1.0)
        
        # Output projection
        self.proj_conv = QConv2D(c1 // self.Q, c2 // self.Q, kernel_size=3, stride=1, padding=1)
        self.bn_proj = IQBN(c2 // self.Q)
        self.act_proj = QReLU()
        
        self.gc = gc
    
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == self.Q, f"Quaternion dimension must be {self.Q}"
        
        # Reshape for convolutions: (B, C, Q, H, W) -> (B*Q, C//Q, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C // Q, H, W)
        
        # Apply embedding convolution if defined
        if self.ec_conv is not None:
            embed = self.ec_conv(x_reshaped)
            embed = self.bn_ec(embed)
            embed = self.act_ec(embed)
        else:
            embed = x_reshaped
        
        # Process guide tensor
        guide = self.gl(guide)  # [B, ec]
        
        # Reshape embed for attention
        embed = embed.view(B, Q, self.nh, -1, H, W)  # [B, Q, nh, hc, H, W]
        
        # Compute attention scores
        embed_flat = embed.view(B, Q, self.nh, -1)  # [B, Q, nh, hc*H*W]
        guide_flat = guide.view(B, 1, -1)  # [B, 1, ec]
        
        # Compute attention with quaternion components
        attn = torch.einsum('bqnc,bc->bqn', embed_flat, guide_flat)  # [B, Q, nh]
        attn = attn.sigmoid() * self.scale
        
        # Apply attention
        embed = embed * attn.view(B, Q, self.nh, 1, 1, 1)
        embed = embed.view(B * Q, -1, H, W)
        
        # Project output
        out = self.proj_conv(embed)
        out = self.bn_proj(out)
        out = self.act_proj(out)
        
        # Reshape back to quaternion format
        out = out.view(B, Q, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()
        
        return out

class C2PSA(nn.Module):
    """C2PSA module with proper quaternion handling."""
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5):
        super(C2PSA, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4 for quaternions."
        
        self.Q = 4
        self.c = int(out_channels * e)
        self.c = (self.c // 4) * 4  # Ensure multiple of 4
        
        # Quaternion-aware convolution to reduce channels
        self.cv1 = QConv2D(in_channels, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # PSABlock instances stored in a ModuleList for dynamic processing
        self.m = nn.ModuleList([
            PSABlock(c=self.c, attn_ratio=1.0, num_heads=self.c // (4 * 4), shortcut=True) for _ in range(n)
        ])
        
        # Attention block
        self.attn = MaxSigmoidAttnBlock(c1=2 * self.c, c2=self.c, nh=8, ec=128, gc=512, scale=True)
        
        # Final convolution to restore channels
        self.cv2 = QConv2D((2 + n + 1) * self.c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(out_channels)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        # Initial convolution, normalization, and activation
        y = self.cv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        
        # Split into 'a' and 'b'
        y = y.chunk(2, dim=1)  # Each has channels = self.c
        
        # Pass through PSABlock instances
        for m in self.m:
            y.append(m(y[-1]))
        
        # Apply attention to the last PSABlock output
        y.append(self.attn(y[-1], guide))
        
        # Concatenate all features
        y = torch.cat(y, dim=1)  # Shape: (B, (2 + n + 1) * self.c, 4, H, W)
        
        # Final convolution, normalization, and activation
        y = self.cv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        
        return y

   
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

class QEnhancedDetectHead(nn.Module):
    """
    Enhanced detection head with improved feature utilization
    and quaternion structure preservation.
    """
    def __init__(self, nc: int, ch: List[int]):
        super().__init__()
        
        # Feature processing branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                QAdaptiveFeatureExtraction(c),
                QDualAttention(c),
                QConv2D(c, c, 3, padding=1),
                IQBN(c),
                QReLU()
            ) for c in ch
        ])
        
        # Classification head
        self.cls_convs = nn.ModuleList([
            QConv2D(c, nc, 1) for c in ch
        ])
        
        # Box regression head
        self.reg_convs = nn.ModuleList([
            QConv2D(c, 4, 1) for c in ch
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps [P3, P4, P5]
        Returns:
            cls_preds: List of classification predictions
            reg_preds: List of box regression predictions
        """
        cls_preds = []
        reg_preds = []
        
        for feat, branch, cls_conv, reg_conv in zip(
            features, self.branches, self.cls_convs, self.reg_convs):
            
            # Process features
            feat = branch(feat)
            
            # Generate predictions
            cls_pred = cls_conv(feat)
            reg_pred = reg_conv(feat)
            
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
        
        return cls_preds, reg_preds


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
