# block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1d, QConv2d, QConv3d
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN

__all__ = (
    "QC1",
    "QC2",
    "QC2",
    "SPPF",
    "C2f",
    "C2fAttn",
    "QBottleneck",
    "QBottleneckCSP",
    "ADown",
    "AConv",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
)

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
    Quaternion Polar Pooling (QPP) layer.
    Emphasizes interchannel relationships by pooling magnitudes and preserving phase information.
    """
    def __init__(self, kernel_size, stride=None, padding=0, mode='max'):
        """
        Initializes the Quaternion Polar Pooling layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window.
            stride (int or tuple, optional): Stride of the pooling window. Defaults to kernel_size.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
            mode (str): Pooling mode - 'max' or 'average'.
        """
        super(QuaternionPolarPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        if mode not in ['max', 'average']:
            raise ValueError("Mode must be 'max' or 'average'")
        self.mode = mode

    def forward(self, x):
        """
        Forward pass for Quaternion Polar Pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, 4, H, W).

        Returns:
            torch.Tensor: Pooled tensor with preserved quaternion structure.
        """
        # Ensure channel dimension is a multiple of 4
        batch_size, channels, quat_dim, H, W = x.shape
        assert quat_dim == 4, "Quaternion dimension must be 4."
        assert channels % 4 == 0, "Number of channels must be a multiple of 4."

        # Reshape to separate quaternion components
        x = x.view(batch_size, channels // 4, 4, H, W)  # Shape: (B, C_q, 4, H, W)

        # Compute magnitudes
        magnitudes = torch.norm(x, dim=2, keepdim=True)  # Shape: (B, C_q, 1, H, W)

        if self.mode == 'max':
            pooled_magnitudes = F.max_pool2d(magnitudes, self.kernel_size, self.stride, self.padding)
        elif self.mode == 'average':
            pooled_magnitudes = F.avg_pool2d(magnitudes, self.kernel_size, self.stride, self.padding)

        # Compute phases (angles) for each component
        phases = torch.atan2(x, torch.clamp(x, min=1e-8))  # Shape: (B, C_q, 4, H, W)

        # Aggregate phases using average pooling to preserve interchannel relationships
        pooled_phases = F.avg_pool2d(phases, self.kernel_size, self.stride, self.padding)

        # Reconstruct pooled quaternion
        pooled_x = pooled_magnitudes * torch.cos(pooled_phases) + \
                   pooled_magnitudes * torch.sin(pooled_phases) * torch.tensor([1, 1, 1, 1], device=x.device).view(1, 1, 4, 1, 1)

        # Reshape back to original channel format
        pooled_x = pooled_x.view(batch_size, -1, 4, pooled_x.shape[-2], pooled_x.shape[-1])

        return pooled_x

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
        self.cv1 = QConv2d(in_channels, c_, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Max pooling with kernel size and stride of 1
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Final convolution after concatenation to project back to out_channels
        self.cv2 = QConv2d(c_ * 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SPPF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, 4, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, 4, H, W).
        """
        # Initial convolution
        y = self.cv1(x)  # Shape: (B, c_, 4, H, W)
        
        # Apply three max pooling layers and concatenate outputs
        pooled_outputs = [y]  # Initialize with first convolution output
        for _ in range(3):
            pooled_outputs.append(self.m(pooled_outputs[-1]))  # Max pooling
        
        # Concatenate along channel dimension and apply final convolution
        y = torch.cat(pooled_outputs, dim=1)  # Shape: (B, 4 * c_, 4, H, W)
        y = self.cv2(y)  # Project to out_channels: (B, out_channels, 4, H, W)
        
        return y

# Quaternion CSP Bottleneck (QC1)
class QC1(nn.Module):
    """Quaternion CSP Bottleneck with 1 Convolution"""
    def __init__(self, in_channels, out_channels, n=1):
        super(QC1, self).__init__()
        # Ensure channels are multiples of 4
        out_channels = (out_channels // 4) * 4
        in_channels = (in_channels // 4) * 4

        self.cv1 = QConv2d(rank=2, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = QBN(out_channels)
        self.act1 = QuaternionActivation(nn.SiLU())

        self.m = nn.Sequential(*[
            QBottleneck(out_channels, out_channels, shortcut=True, g=1, expansion=0.5) for _ in range(n)
        ])

    def forward(self, x):
        y = self.cv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.m(y)
        return y + y  # Residual connection

class QC2(nn.Module):
    """
    Quaternion CSP Bottleneck with 2 Quaternion Convolutions.
    
    This module splits the input into two parts, processes one part through a series of quaternion bottleneck blocks,
    and then concatenates it back with the other part before projecting to the desired number of output channels.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        n (int): Number of quaternion bottleneck blocks.
        shortcut (bool): Whether to include residual connections.
        g (int): Number of groups for convolution.
        e (float): Expansion ratio for hidden channels.
    """
    
    def __init__(self, c1: int, c2: int, n: int =1, shortcut: bool = True, g: int =1, e: float=0.5):
        super(QC2, self).__init__()
        assert c1 %4 ==0 and c2 %4 ==0, "Channels must be multiples of 4 for quaternions."
        self.c = int(c2 * e)
        self.c = (self.c //4 ) *4  # Ensure multiple of 4
        
        # Initial convolution to reduce channels
        self.cv1 = QConv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # Sequential Quaternion Bottleneck blocks
        self.m = nn.Sequential(*[
            QBottleneck(in_channels=self.c, out_channels=self.c, shortcut=shortcut, groups=g, expansion=1.0) for _ in range(n)
        ])
        
        # Final convolution to restore channels
        self.cv2 = QConv2d(2 * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QC2 module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H, W).
        """
        # Initial convolution
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Split into two parts
        a, b = x.chunk(2, dim=1)  # Each has channels = self.c
        
        # Process 'a' through quaternion bottleneck blocks
        a = self.m(a)
        
        # Concatenate 'a' and 'b'
        out = torch.cat((a, b), dim=1)  # Shape: (B, 2*c, 4, H, W)
        
        # Final convolution
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        return out

# Quaternion CSP C3 Variant
class QC3(nn.Module):
    """Quaternion CSP Bottleneck with 3 Convolutions"""
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super(QC3, self).__init__()
        # Ensure channels are multiples of 4
        in_channels = (in_channels // 4) * 4
        out_channels = (out_channels // 4) * 4
        hidden_channels = int(out_channels * expansion)
        hidden_channels = (hidden_channels // 4) * 4

        self.cv1 = QConv2d(rank=2, in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = QBN(hidden_channels)
        self.act1 = QuaternionActivation(nn.SiLU())

        self.m = nn.Sequential(*[
            QBottleneck(hidden_channels, hidden_channels, shortcut=shortcut, g=groups, expansion=expansion) for _ in range(n)
        ])

        self.cv2 = QConv2d(rank=2, in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = QBN(hidden_channels)
        self.act2 = QuaternionActivation(nn.SiLU())

        self.cv3 = QConv2d(rank=2, in_channels=2 * hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = QBN(out_channels)
        self.act3 = QuaternionActivation(nn.SiLU())

    def forward(self, x):
        y1 = self.cv1(x)
        y1 = self.bn1(y1)
        y1 = self.act1(y1)
        y1 = self.m(y1)

        y2 = self.cv2(x)
        y2 = self.bn2(y2)
        y2 = self.act2(y2)

        y = torch.cat((y1, y2), dim=1)
        y = self.cv3(y)
        y = self.bn3(y)
        y = self.act3(y)
        return y

class QCSPBottleneck(nn.Module):
    """
    General Quaternion CSP Bottleneck.
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, groups=1, expansion=0.5):
        super(QCSPBottleneck, self).__init__()
        # Ensure channels are multiples of 4
        in_channels = (in_channels // 4) * 4
        out_channels = (out_channels // 4) * 4
        hidden_channels = int(out_channels * expansion)
        hidden_channels = (hidden_channels // 4) * 4

        self.cv1 = QConv2d(rank=2, in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = QBN(hidden_channels)
        self.act1 = QuaternionActivation(nn.SiLU())

        self.m = nn.Sequential(*[
            QBottleneck(hidden_channels, hidden_channels, shortcut=shortcut, g=groups, expansion=expansion) for _ in range(n)
        ])

        self.cv2 = QConv2d(rank=2, in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = QBN(hidden_channels)
        self.act2 = QuaternionActivation(nn.SiLU())

        self.cv3 = QConv2d(rank=2, in_channels=2 * hidden_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = QBN(out_channels)
        self.act3 = QuaternionActivation(nn.SiLU())

    def forward(self, x):
        y1 = self.cv1(x)
        y1 = self.bn1(y1)
        y1 = self.act1(y1)
        y1 = self.m(y1)

        y2 = self.cv2(x)
        y2 = self.bn2(y2)
        y2 = self.act2(y2)

        y = torch.cat((y1, y2), dim=1)
        y = self.cv3(y)
        y = self.bn3(y)
        y = self.act3(y)
        return y

class AConv(nn.Module):
    """
    Quaternion Attention Convolution (AConv).
    
    This module applies an average pooling followed by a quaternion convolution.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
    """
    def __init__(self, c1: int, c2: int):
        super(AConv, self).__init__()
        assert c1 %4 ==0 and c2 %4 ==0, "Channels must be multiples of 4 for quaternions."
        
        # Average pooling followed by quaternion convolution
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False)
        self.conv = QConv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = IQBN(c2)
        self.act = QReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the AConv module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H/2, W/2).
        """
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class ADown(nn.Module):
    """
    Quaternion Attention Downsampling (ADown).
    
    This module downsamples the input tensor using a combination of average pooling, max pooling, and quaternion convolutions.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
    """
    def __init__(self, c1: int, c2: int):
        super(ADown, self).__init__()
        assert c1 %4 ==0 and c2 %4 ==0, "Channels must be multiples of 4 for quaternions."
        
        self.c = c2 // 2
        assert self.c %4 ==0, "Intermediate channels must be multiples of 4 for quaternions."
        
        # First branch: Average pooling followed by quaternion convolution
        self.conv1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False),
            QConv2d(c1 // 2, self.c, kernel_size=3, stride=1, padding=1, bias=False),
            IQBN(self.c),
            QReLU()
        )
        
        # Second branch: Max pooling followed by quaternion convolution
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            QConv2d(c1 // 2, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c),
            QReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ADown module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H/2, W/2).
        """
        # Split the input channels
        x1, x2 = x.chunk(2, dim=1)  # Each has c1//2 channels
        
        # Apply two branches
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        
        # Concatenate along the channel dimension
        out = torch.cat((y1, y2), dim=1)  # Shape: (B, C2, 4, H/2, W/2)
        
        return out

class CBLinear(nn.Module):
    """
    Quaternion Channel-Balanced Linear Layer (CBLinear).
    
    Applies a quaternion convolution followed by splitting the output into specified channel sizes.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2s (list or tuple): List of output channel sizes (each must be a multiple of 4).
        k (int): Kernel size for convolution. Defaults to 1.
        s (int): Stride for convolution. Defaults to 1.
        p (int): Padding for convolution. If None, auto-padding is applied based on kernel size.
        g (int): Number of groups for convolution. Defaults to 1.
    """
    def __init__(self, c1: int, c2s: list, k: int =1, s: int =1, p: Optional[int] =None, g: int =1):
        super(CBLinear, self).__init__()
        assert c1 %4 ==0 and all(c2 %4 ==0 for c2 in c2s), "Channels must be multiples of 4 for quaternions."
        
        total_c2 = sum(c2s)
        self.c2s = c2s
        padding = autopad(k, p)  # Implement autopad function as needed
        
        self.conv = QConv2d(c1, total_c2, kernel_size=k, stride=s, padding=padding, groups=g, bias=True)
    
    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through the CBLinear module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            list: List of output tensors split according to c2s.
        """
        out = self.conv(x)  # Shape: (B, sum(c2s), 4, H_out, W_out)
        out_splits = out.split(self.c2s, dim=1)  # List of tensors
        return out_splits


class CBFuse(nn.Module):
    """
    Quaternion Channel-Balanced Fuse (CBFuse).
    
    This module fuses features from multiple sources by interpolating to a target size and summing them.
    
    Args:
        idx (list): List of indices indicating which inputs to fuse.
    """
    def __init__(self, idx: list):
        super(CBFuse, self).__init__()
        self.idx = idx
    
    def forward(self, xs: list) -> torch.Tensor:
        """
        Forward pass through the CBFuse module.
        
        Args:
            xs (list): List of input tensors. Each tensor should have shape (B, C, 4, H, W).
        
        Returns:
            torch.Tensor: Fused tensor of shape matching the last input's spatial dimensions.
        """
        target_size = xs[-1].shape[-2:]  # (H, W)
        fused = 0
        for i, x in enumerate(xs[:-1]):
            # Interpolate each input to the target size
            interpolated = F.interpolate(x[self.idx[i]], size=target_size, mode="nearest")
            fused += interpolated
        # Add the last input without modification
        fused += xs[-1]
        return fused

class QBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=1.0):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        hidden_channels = max(4, (hidden_channels // 4) * 4)  # Ensure multiple of 4 and at least 4

        self.cv1 = QConv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = QConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.cv1(x)
        out = self.cv2(out)
        if self.shortcut:
            out = out + x
        return out

# 

class C3k2(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, e=0.5, c3k=False, shortcut=True):
        super().__init__()
        hidden_channels = int(out_channels * e)
        hidden_channels = max(4, (hidden_channels // 4) * 4)  # Ensure multiple of 4 and at least 4

        self.cv1 = QConv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = QConv2d(in_channels, hidden_channels, kernel_size=1, stride=1)

        self.m = nn.Sequential(
            *[QBottleneck(hidden_channels, hidden_channels, shortcut=shortcut, expansion=1.0) for _ in range(n)]
        )

        self.cv3 = QConv2d(2 * hidden_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # print(f"C3k2 Input: {x.shape}")
        y1 = self.cv1(x)
        # print(f"After cv1: {y1.shape}")
        y1 = self.m(y1)
        # print(f"After m: {y1.shape}")
        y2 = self.cv2(x)
        # print(f"After cv2: {y2.shape}")
        y = torch.cat((y1, y2), dim=1)
        # print(f"After concat: {y.shape}")
        out = self.cv3(y)
        # print(f"C3k2 Output: {out.shape}")
        return out

class C3k(nn.Module):
    """
    Quaternion C3k Module.
    
    Applies a series of Quaternion Bottleneck blocks followed by a convolution with customizable kernel size.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        n (int): Number of bottleneck blocks to stack.
        shortcut (bool): Whether to include residual connections.
        g (int): Number of groups for convolution.
        e (float): Expansion ratio for hidden channels.
        k (int): Kernel size for the final convolution.
    """
    def __init__(self, c1: int, c2: int, n: int =1, shortcut: bool =True, g: int =1, e: float=0.5, k: int=3):
        super(C3k, self).__init__()
        assert c1 %4 ==0 and c2 %4 ==0, "Channels must be multiples of 4 for quaternions."
        self.c = int(c2 * e)
        self.c = (self.c //4 ) *4  # Ensure multiple of 4
        
        # Initial convolution
        self.cv1 = QConv2d(c1, self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(self.c)
        self.act1 = QReLU()
        
        # Sequential Bottleneck blocks with customizable kernel sizes
        self.m = nn.Sequential(*[
            QBottleneck(in_channels=self.c, out_channels=self.c, shortcut=shortcut, groups=g, expansion=1.0) for _ in range(n)
        ])
        
        # Final convolution with customizable kernel size
        self.cv2 = QConv2d(self.c, c2, kernel_size=k, stride=1, padding=k//2, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the C3k module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H, W).
        """
        # Initial convolution
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Apply sequential bottleneck blocks
        x = self.m(x)
        
        # Final convolution
        x = self.cv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape  # Example: [4, -1, H, W]

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
        
        self.dim = dim  # Add this line
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        
        # Total output channels for qkv (query, key, value)
        nh_kd = self.num_heads * self.key_dim
        h = nh_kd * 3  # For q, k, v
        
        # Quaternion-aware convolutions
        self.qkv = QConv2d(dim, h, kernel_size=1, stride=1)
        self.proj = QConv2d(dim, dim, kernel_size=1, stride=1)
        self.pe = QConv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        
        # Initialize weights
        for layer in [self.qkv, self.proj, self.pe]:
            if isinstance(layer, QConv2d):
                nn.init.normal_(layer.modulus, std=0.02)
                nn.init.constant_(layer.phase, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Quaternion Attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, Q, H, W)
        
        Returns:
            torch.Tensor: Output tensor after self-attention, shape (B, C, Q, H, W)
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Quaternion dimension must be 4."
        assert C % self.num_heads == 0, "Channels must be divisible by num_heads."
        
        # Reshape to combine batch and quaternion dimensions
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)  # [B*4, C, H, W]
        
        # Compute Q, K, V
        qkv = self.qkv(x_flat)  # [B*4, 3*nh_kd, H, W]
        qkv = qkv.chunk(3, dim=1)  # [B*4, nh_kd, H, W] each
        q, k, v = qkv
        
        # Reshape for multi-head attention
        q = q.view(B, Q, self.num_heads, self.key_dim, H * W)  # [B, 4, num_heads, key_dim, H*W]
        k = k.view(B, Q, self.num_heads, self.key_dim, H * W)  # [B, 4, num_heads, key_dim, H*W]
        v = v.view(B, Q, self.num_heads, self.head_dim, H * W)  # [B, 4, num_heads, head_dim, H*W]
        
        # Permute to bring num_heads to the front
        q = q.permute(0, 2, 1, 3, 4)  # [B, num_heads, 4, key_dim, H*W]
        k = k.permute(0, 2, 1, 3, 4)  # [B, num_heads, 4, key_dim, H*W]
        v = v.permute(0, 2, 1, 3, 4)  # [B, num_heads, 4, head_dim, H*W]
        
        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, 4, key_dim, key_dim]
        attn = torch.softmax(attn_scores, dim=-1)  # [B, num_heads, 4, key_dim, key_dim]
        
        # Apply attention to V
        out = attn @ v  # [B, num_heads, 4, key_dim, H*W]
        
        # Reshape and permute back
        out = out.permute(0, 2, 1, 3, 4).reshape(B * Q, self.num_heads * self.key_dim, H, W)  # [B*4, C, H, W]
        
        # Apply positional embedding and projection
        out = self.proj(out) + self.pe(x_flat)  # [B*4, C, H, W]
        
        # Reshape back to quaternion structure
        out = out.view(B, Q, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, C, 4, H, W]
        
        return out

class PSABlock(nn.Module):
    """Position Sensitive Attention Block for quaternion features."""
    def __init__(self, c: int, attn_ratio: float = 1.0, num_heads: int = 8, shortcut: bool = True):
        super().__init__()
        assert c % 4 == 0, "Number of channels must be divisible by 4 for quaternions."
        self.Q = 4  # Quaternion dimension

        self.attn = QAttention(dim=c // self.Q, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            QConv2d(c // self.Q, (c * 2) // self.Q, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN((c * 2) // self.Q),
            QReLU(),
            QConv2d((c * 2) // self.Q, c // self.Q, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(c // self.Q)
        )
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PSABlock.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W] or [B, Q, C_per_quaternion, H, W]

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Check tensor dimensions and reshape if necessary
        if x.dim() == 4:
            B, C, H, W = x.shape
            Q = self.Q
            assert C % Q == 0, f"Channel dimension {C} is not divisible by Q={Q}."
            C_q = C // Q
            x = x.view(B, Q, C_q, H, W)  # Reshape to [B, Q, C_q, H, W]
        elif x.dim() == 5:
            B, Q, C_q, H, W = x.shape
            assert Q == self.Q, f"Expected quaternion dimension {self.Q}, got {Q}."
        else:
            raise ValueError(f"Expected input tensor with 4 or 5 dimensions, got {x.dim()}.")

        # Permute to [B, C_q, Q, H, W] for QAttention
        x_permuted = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C_q, Q, H, W]

        # Apply attention
        out = self.attn(x_permuted)  # [B, C_q, Q, H, W]
        if self.shortcut:
            out = out + x_permuted  # Residual connection

        # Reshape for FFN: [B*Q, C_q, H, W]
        out_reshaped = out.permute(0, 2, 1, 3, 4).reshape(B * Q, C_q, H, W)

        # Apply FFN
        ffn_out = self.ffn(out_reshaped)  # [B*Q, C_q, H, W]
        if self.shortcut:
            ffn_out = ffn_out + out_reshaped  # Residual connection

        # Reshape back to [B, C_q, Q, H, W]
        out_final = ffn_out.view(B, Q, C_q, H, W)

        # Permute back to [B, Q, C_q, H, W]
        return out_final

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
        self.cv1 = QConv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # Split into two parts: 'a' and 'b'
        # 'a' will go through attention and FFN
        # 'b' will remain unchanged
        # Apply attention to 'a'
        self.attn = QAttention(dim=self.c, num_heads=self.c // (4 * 4), attn_ratio=1.0)
        
        # Feed-Forward Network for 'a'
        self.ffn = nn.Sequential(
            QConv2d(self.c, self.c * 2, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c * 2),
            QReLU(),
            QConv2d(self.c * 2, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c)
        )
        
        # Final convolution to restore channels
        self.cv2 = QConv2d(2 * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H, W).
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
        out = torch.cat((a, b), dim=1)  # Shape: (B, 2 * self.c, 4, H, W)
        
        # Final convolution, normalization, and activation
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        return out

# Fixed MaxSigmoidAttnBlock
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
            self.ec_conv = QConv2d(c1 // self.Q, ec // self.Q, kernel_size=1, stride=1)
            self.bn_ec = VQBN(ec // self.Q)
            self.act_ec = QReLU()
        else:
            self.ec_conv = None
        
        # Guide processing
        self.gl = nn.Linear(gc, ec)
        
        # Attention parameters
        self.bias = nn.Parameter(torch.zeros(nh))
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else torch.tensor(1.0)
        
        # Output projection
        self.proj_conv = QConv2d(c1 // self.Q, c2 // self.Q, kernel_size=3, stride=1, padding=1)
        self.bn_proj = VQBN(c2 // self.Q)
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
        out = out.view(B, Q, -1, H, W).permute(0, 2, 1, 3, 4)
        
        return out


class C2PSA(nn.Module):
    """C2PSA module with proper quaternion handling."""
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4"
        
        self.Q = 4
        self.c = int(out_channels * e)
        self.c = (self.c // 4) * 4  # Ensure multiple of 4
        
        # Initial convolution
        self.cv1 = QConv2d(in_channels//self.Q, (2 * self.c)//self.Q, kernel_size=1)
        self.bn1 = VQBN((2 * self.c)//self.Q)
        self.act1 = QReLU()
        
        # PSA blocks
        self.m = nn.Sequential(*[
            PSABlock(
                c=self.c,
                attn_ratio=1.0,
                num_heads=max(1, self.c//(4*16)),
                shortcut=True
            ) for _ in range(n)
        ])
        
        # Final convolution
        self.cv2 = QConv2d((2 * self.c)//self.Q, out_channels//self.Q, kernel_size=1)
        self.bn2 = VQBN(out_channels//self.Q)
        self.act2 = QReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == self.Q, f"Quaternion dimension must be {self.Q}"
        
        # Reshape before permute
        x = x.view(B, C // Q, Q, H, W)  # [B, C//Q, Q, H, W]
        
        # Permute to [B, Q, C//Q, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, Q, C//Q, H, W]
        
        # Reshape for initial convolution
        x = x.reshape(B * Q, C // Q, H, W)  # [B*Q, C//Q, H, W]
        
        # Initial processing
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Reshape back to quaternion format for PSA blocks
        x = x.view(B, Q, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, 2*c//Q, Q, H, W]
        
        # Split channels
        a, b = x.chunk(2, dim=1)  # Each [B, c//Q, Q, H, W]
        
        # Process through PSA blocks
        b = self.m(b)  # [B, c//Q, Q, H, W]
        
        # Concatenate channels
        x = torch.cat([a, b], dim=1)  # [B, 2*c//Q, Q, H, W]
        
        # Reshape for final convolution
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Q, -1, H, W)  # [B*Q, 2*c//Q, H, W]
        
        # Final processing
        x = self.cv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Reshape back to quaternion format
        x = x.view(B, Q, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, out_channels//Q, Q, H, W]
        
        return x


class C2fPSA(nn.Module):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks in quaternion neural networks.
    
    Extends the C2f module by incorporating multiple PSABlock instances for improved attention mechanisms.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        n (int): Number of PSABlock instances to stack.
        e (float): Expansion ratio for hidden channels.
        g (int): Number of groups for grouped convolutions.
        shortcut (bool): Whether to include residual connections.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super(C2fPSA, self).__init__()
        assert c1 % 4 == 0 and c2 % 4 == 0, "Input and output channels must be multiples of 4."
        
        self.c = int(c2 * e)
        self.c = (self.c // 4) * 4  # Ensure hidden channels are multiples of 4
        assert self.c > 0, "Hidden channels must be positive and a multiple of 4."
        
        # Quaternion-aware convolution to reduce channels
        self.cv1 = QConv2d(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # PSABlock instances stored in a ModuleList for dynamic processing
        self.m = nn.ModuleList([
            PSABlock(c=self.c, attn_ratio=1.0, num_heads=self.c // (4 * 4), shortcut=True) for _ in range(n)
        ])
        
        # Attention block
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)
        
        # Final convolution to restore channels
        self.cv2 = QConv2d((2 + n + 1) * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C2fPSA layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
            guide (torch.Tensor): Guide tensor for attention mechanism.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H, W).
        """
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

