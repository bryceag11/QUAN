# block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1D, QConv2D, QConv3D, Conv
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
from typing import List
import math


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

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

class QuaternionMaxPool(nn.Module):
    """Quaternion-aware max pooling"""
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        pooled = self.pool(x_reshaped)
        
        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

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
    Spatial Pyramid Pooling - Fast (SPPF) layer for quaternion neural networks.
    Maintains quaternion structure throughout the pooling pyramid.
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        
        # Set up intermediate channels (half of in_channels), ensuring multiple of 4
        c_ = c1 // 2
        assert c_ % 4 == 0, "Hidden channels must be a multiple of 4"

        # First convolution to reduce channel dimensionality
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Max pooling with kernel size and stride of 1
        self.m = QuaternionMaxPool(kernel_size=k, stride=1, padding = k//2)
        
        # Final convolution after concatenation to project back to out_channels
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

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


class C3k2(nn.Module):
    """
    Enhanced C3k2 with parallel processing paths for quaternion features.
    Maintains cross-stage partial design while preserving quaternion structure.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.m = nn.Sequential(*[
            C3k(self.c, self.c, 2, shortcut, g) if c3k else 
            QBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        ])
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel processing.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        # Split processing
        y = list(self.cv1(x).chunk(2, 1))
        
        # Extend with outputs from each module in self.m
        y.extend(m(y[-1]) for m in self.m)
        
        # Combine all paths
        return self.cv2(torch.cat(y, 1))
    

class C3k(nn.Module):
    """
    C3k module for quaternion data - CSP bottleneck with customizable kernel sizes.
    This version is designed to work with the parallel C3k2 implementation.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2*e)        
        # Calculate hidden channels (ensure multiple of 4)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        
        # Bottleneck block sequence
        self.m = nn.Sequential(*(QBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass maintaining quaternion structure.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        # Store identity if using shortcut
        # identity = x if self.shortcut else None
        
        # # Initial convolution
        # x = self.cv1(x)
        
        # # Process through bottlenecks
        # x = self.m(x)
        
        # # Final convolution
        # x = self.cv2(x)
        
        # # Add shortcut if enabled
        # if self.shortcut:
        #     x = x + identity
            
        
        # return x
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class QBottleneck(nn.Module):
    """
    Quaternion-aware bottleneck block used in C3k.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k =(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        
        # First 1x1 conv to reduce channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quaternion bottleneck.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        identity = x if self.add else None
        
        # First conv + bn + relu
        x = self.cv1(x)

        
        # Second conv + bn + relu
        x = self.cv2(x)
        
        # Add shortcut if enabled
        if self.add:
            x = x + identity
                
        return x


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
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

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
        dim (int): The input quaternion channels
        num_heads (int): Number of attention heads
        attn_ratio (float): Ratio to determine key dimension
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        
        # Ensure dimensions are properly calculated
        # Calculate dimensions ensuring they're compatible
        self.head_dim = max(dim // num_heads, 1)
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # Calculate total dimensions for q, k, v
        self.qk_dim = self.key_dim * num_heads  # For both query and key
        self.v_dim = self.head_dim * num_heads  # For value
        self.total_dim = self.qk_dim * 2 + self.v_dim  # Total QKV dimension
        
        # Quaternion-aware layers
        self.qkv = QConv2D(dim, self.total_dim, kernel_size=1)
        self.proj = QConv2D(dim, dim, kernel_size=1)
        self.pe = QConv2D(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim//4)
        
        # Layer normalization
        self.norm = IQBN(dim)
        
        # Store split sizes
        self.split_sizes = [self.qk_dim//4, self.qk_dim//4, self.v_dim//4]
        print(f"Attention dimensions:")
        print(f"Input dim: {dim}")
        print(f"Head dim: {self.head_dim}")
        print(f"Key dim: {self.key_dim}")
        print(f"QK dim: {self.qk_dim}")
        print(f"V dim: {self.v_dim}")
        print(f"Total dim: {self.total_dim}")
        print(f"Split sizes: {self.split_sizes}")

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass preserving quaternion structure.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input (Q=4)"
        N = H * W
        
        # Debug prints
        print(f"\nInput shape: {x.shape}")
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        print(f"After QKV projection: {qkv.shape}")
        qkv = qkv.reshape(B, -1, Q, H * W)
        print(f"After QKV reshape: {qkv.shape}")

        # Split into query, key, value
        q, k, v = qkv.split(self.split_sizes, dim=1)
        # Reshape to prepare for splitting
        q = q.reshape(B, self.num_heads, -1, Q, N)
        k = k.reshape(B, self.num_heads, -1, Q, N)
        v = v.reshape(B, self.num_heads, -1, Q, N)
        print(f"After reshape: {q.shape}")

        # Rearrange for attention computation
        q = q.permute(0, 1, 3, 4, 2)  # [B, heads, Q, key_dim, N]
        k = k.permute(0, 1, 3, 2, 4)  # [B, heads, Q, N, key_dim]
        v = v.permute(0, 1, 3, 4, 2)  # [B, heads, Q, N, head_dim]
        
        # Compute attention
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        x = torch.matmul(attn, v)  # [B, heads, Q, N, head_dim]
        
        # Reshape back
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C, Q, H, W)
        
        # Final processing
        pe = self.pe(x)
        x = x + pe
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
    def __init__(self, c1, c2, n=1, e=0.5):
        super(C2PSA, self).__init__()
        

        
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)

        
        # PSA blocks
        self.m = nn.Sequential(*[
            PSABlock(
                c=self.c,  # This is already in quaternion channels
                attn_ratio=0.5,
                num_heads=max(1, self.c // 16)  # Adjust head count appropriately
            ) for _ in range(n)
        ])
        
    def forward(self, x):
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

   
