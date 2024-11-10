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


class QBottleneck(nn.Module):
    """
    Quaternion Bottleneck block that preserves quaternion properties.
    Ensures channel dimensions are multiples of 4.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super(QBottleneck, self).__init__()
        hidden_channels = int(out_channels * expansion)
        # Ensure hidden_channels is a multiple of 4
        hidden_channels = (hidden_channels // 4) * 4

        self.cv1 = QConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = QBN(hidden_channels)

        self.act1 = QuaternionActivation(nn.SiLU())

        self.cv2 = QConv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=groups,
            bias=False
        )
        self.bn2 = QBN(hidden_channels)
        self.act2 = QuaternionActivation(nn.SiLU())

        self.cv3 = QConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = QBN(out_channels)

        self.shortcut = shortcut and (in_channels == out_channels)
        if self.shortcut:
            self.bn_skip = QBN(out_channels)

    def forward(self, x):
        identity = x

        out = self.cv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = self.cv3(out)
        out = self.bn3(out)

        if self.shortcut:
            identity = self.bn_skip(identity)
            out += identity

        out = QuaternionActivation(nn.SiLU())(out)
        return out

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

class C3k2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5, g: int = 1, shortcut: bool = True, **kwargs):
        """
        Initializes the C3k2 module with specified channels and configurations.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            n (int): Number of repeats for the module.
            e (float): Expansion rate or ratio.
            g (int): Group convolution factor.
            shortcut (bool): Whether to include a shortcut connection.
            **kwargs: Additional arguments for flexibility, ignored if unused.
        """
        super(C3k2, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4 for quaternions."
        self.c = int(out_channels * e)
        self.c = (self.c // 4) * 4  # Ensure multiple of 4

        # Initial convolutions
        self.cv1 = QConv2d(in_channels, self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.cv2 = QConv2d(in_channels, self.c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sequential blocks: either C3k or QBottleneck based on kwargs
        self.m = nn.Sequential(*[
            QBottleneck(in_channels=self.c, out_channels=self.c, shortcut=shortcut, groups=g, expansion=1.0) for _ in range(n)
        ])

        # Final convolution
        self.cv3 = QConv2d((2 + n) * self.c, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = IQBN(out_channels)
        self.act3 = QReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the C3k2 module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, 4, H, W).
        """
        # Initial convolutions
        y = [self.cv2(x), self.cv1(x)]  # y[0] = cv2(x), y[1] = cv1(x)
        
        # Apply sequential blocks
        for m in self.m:
            y.append(m(y[-1]))
        
        # Concatenate all processed features
        out = torch.cat(y, dim=1)  # Shape: (B, (2 + n) * c, 4, H, W)
        
        # Final convolution
        out = self.cv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        
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

class MaxSigmoidAttnBlock(nn.Module):
    """
    Quaternion Max Sigmoid Attention Block.
    
    This module applies a max-sigmoid attention mechanism to enhance feature representation by emphasizing important regions.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        nh (int): Number of heads.
        ec (int): Embedding channels.
        gc (int): Global channels.
        scale (bool): Whether to apply scaling.
    """
    def __init__(self, c1: int, c2: int, nh: int =1, ec: int =128, gc: int =512, scale: bool =False):
        super(MaxSigmoidAttnBlock, self).__init__()
        assert c1 %4 ==0 and c2 %4 ==0, "Channels must be multiples of 4 for quaternions."
        
        self.nh = nh
        self.hc = c2 // nh
        assert self.hc %4 ==0, "h*c must be a multiple of 4 for quaternions."
        
        # Optional embedding convolution
        self.ec = QConv2d(c1, ec, kernel_size=1, stride=1, padding=0, bias=False) if c1 != ec else None
        self.bn_ec = IQBN(ec) if self.ec is not None else None
        self.act_ec = QReLU() if self.ec is not None else None
        
        # Linear layer for global context
        self.gl = nn.Linear(gc, ec)
        
        # Bias parameter
        self.bias = nn.Parameter(torch.zeros(nh))
        
        # Projection convolution
        self.proj_conv = QConv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_proj = IQBN(c2)
        self.act_proj = QReLU()
        
        # Scaling parameter
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else torch.tensor(1.0)
    
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MaxSigmoidAttnBlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, 4, H, W).
            guide (torch.Tensor): Guide tensor for attention mechanism, shape (B, gc).
        
        Returns:
            torch.Tensor: Output tensor after applying attention and projection.
        """
        B, C, quat_dim, H, W = x.shape
        assert quat_dim ==4, "Quaternion dimension must be 4."
        
        # Apply embedding convolution if defined
        if self.ec is not None:
            embed = self.ec(x)
            embed = self.bn_ec(embed)
            embed = self.act_ec(embed)
        else:
            embed = x  # Shape: (B, C1, 4, H, W)
        
        # Reshape embed for attention computation
        embed = embed.view(B, self.nh, self.hc, H, W)  # Shape: (B, nh, hc, H, W)
        
        # Process guide tensor through linear layer
        guide = self.gl(guide)  # Shape: (B, ec)
        guide = guide.view(B, -1, 1, 1)  # Shape: (B, ec, 1, 1)
        
        # Compute attention scores using einsum for batch matrix multiplication
        # Example: using dot product between embed and guide
        attn_scores = torch.einsum('bnc,nc->bn', embed.view(B, self.nh, -1), guide.squeeze(-1))  # Shape: (B, nh)
        
        # Apply max over spatial dimensions (simplified)
        attn = attn_scores.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, nh, 1, 1)
        attn = torch.sigmoid(attn + self.bias.view(1, -1, 1, 1))  # Shape: (B, nh, 1, 1)
        
        # Apply scaling
        attn = attn * self.scale
        
        # Apply attention to embed
        out = embed * attn.unsqueeze(2)  # Shape: (B, nh, hc, H, W)
        out = out.view(B, -1, H, W)  # Shape: (B, c2, H, W)
        
        # Apply projection convolution
        out = self.proj_conv(out)
        out = self.bn_proj(out)
        out = self.act_proj(out)
        
        return out

class QAttention(nn.Module):
    """
    Quaternion Attention module performing self-attention on quaternion-structured input tensors.
    
    Args:
        dim (int): Number of input channels (must be a multiple of 4).
        num_heads (int): Number of attention heads (dim must be divisible by (num_heads * 4)).
        attn_ratio (float): Ratio to determine key dimension relative to head dimension.
        
    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        key_dim (int): Dimension of the attention key.
        scale (float): Scaling factor for attention scores.
        qkv (QConv2d): Convolutional layer to compute queries, keys, and values.
        proj (QConv2d): Convolutional layer to project the attended values.
        pe (QConv2d): Depthwise convolutional layer for positional encoding.
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 1.0):
        super(QAttention, self).__init__()
        assert dim % (num_heads * 4) == 0, "dim must be divisible by (num_heads * 4)."
        
        self.num_heads = num_heads
        # Each head's dimension should be a multiple of 4 to handle quaternion components
        self.head_dim = (dim // num_heads) // 4 * 4  
        self.key_dim = int(self.head_dim * attn_ratio)
        assert self.key_dim % 4 == 0, "key_dim must be a multiple of 4."
        
        self.scale = self.key_dim ** -0.5
        
        nh_kd = self.key_dim * self.num_heads
        h = dim + nh_kd * 2  # Total output channels for qkv
        
        assert h % 4 == 0, "Total qkv channels (h) must be a multiple of 4."
        
        # Quaternion-aware convolution to compute queries, keys, and values
        self.qkv = QConv2d(dim, h, kernel_size=1, stride=1, padding=0, bias=False)
        # Quaternion-aware convolution to project the attended values back to original dimension
        self.proj = QConv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        # Depthwise convolution for positional encoding
        self.pe = QConv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Quaternion Attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor after self-attention, shape (B, C, 4, H, W).
        """
        B, C, quat_dim, H, W = x.shape
        assert quat_dim == 4, "Quaternion dimension must be 4."
        assert C % 4 == 0, "Number of channels must be a multiple of 4."
        
        # Reshape to (B, C, H, W)
        x = x.view(B, C, H, W)
        
        # Compute queries, keys, and values
        qkv = self.qkv(x)  # Shape: (B, h, H, W)
        h_qkv = qkv.shape[1]
        # Reshape for multi-head attention
        qkv = qkv.view(B, self.num_heads, -1, H * W)  # Shape: (B, num_heads, h_qkv//num_heads, N)
        
        # Split into queries, keys, and values
        split_size = [self.key_dim, self.key_dim, self.head_dim]
        q, k, v = torch.split(qkv, split_size, dim=2)  # Each: (B, num_heads, key_dim or head_dim, N)
        
        # Compute attention scores
        attn = (q.transpose(-2, -1) @ k) * self.scale  # Shape: (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)  # Shape: (B, num_heads, N, N)
        
        # Compute attended values
        out = (v @ attn.transpose(-2, -1))  # Shape: (B, num_heads, head_dim, N)
        out = out.view(B, C, H, W)  # Shape: (B, C, H, W)
        
        # Add positional encoding
        out = out + self.pe(v.view(B, C, H, W))
        
        # Project the output
        out = self.proj(out)  # Shape: (B, C, H, W)
        
        # Reshape back to quaternion structure
        out = out.view(B, C, 4, H, W)
        
        return out

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for quaternion neural networks.
    
    Combines Quaternion Attention with a quaternion-specific feed-forward network and optional residual connections.
    
    Args:
        c (int): Number of input and output channels (must be a multiple of 4).
        attn_ratio (float): Ratio to determine key dimension in attention.
        num_heads (int): Number of attention heads.
        shortcut (bool): Whether to include residual connections.
    """
    def __init__(self, c: int, attn_ratio: float = 1.0, num_heads: int = 8, shortcut: bool = True):
        super(PSABlock, self).__init__()
        assert c % 4 == 0, "Number of channels must be a multiple of 4."
        assert c % (num_heads * 4) == 0, "Number of channels must be divisible by (num_heads * 4)."
        
        self.attn = QAttention(dim=c, num_heads=num_heads, attn_ratio=attn_ratio)
        
        # Feed-Forward Network: Conv -> Activation -> Conv
        self.ffn = nn.Sequential(
            QConv2d(c, c * 2, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(c * 2),
            QReLU(),
            QConv2d(c * 2, c, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(c)
        )
        
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSABlock.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
        
        Returns:
            torch.Tensor: Output tensor after applying attention and feed-forward network.
        """
        residual = x
        out = self.attn(x)
        if self.shortcut:
            out = out + residual
        out = self.ffn(out)
        if self.shortcut:
            out = out + residual
        return out

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


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction in quaternion neural networks.
    
    Args:
        in_channels (int): Number of input channels (must be a multiple of 4).
        out_channels (int): Number of output channels (must be a multiple of 4).
        n (int): Number of PSABlock instances.
        e (float): Expansion ratio for hidden channels.
        g (int): Number of groups for grouped convolutions.
        shortcut (bool): Whether to include residual connections.
    """
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, e: float = 0.5, g: int = 1, shortcut: bool = True):
        super(C2PSA, self).__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, "Channels must be multiples of 4."

        self.hidden_channels = int(out_channels * e)
        self.hidden_channels = (self.hidden_channels // 4) * 4  # Ensure multiple of 4
        assert self.hidden_channels > 0, "Hidden channels must be a positive multiple of 4."

        # Initial quaternion convolution for channel reduction
        self.cv1 = QConv2d(in_channels, 2 * self.hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.hidden_channels)
        self.act1 = QReLU()

        # Stack of PSABlock instances
        self.m = nn.Sequential(*[PSABlock(c=self.hidden_channels, attn_ratio=1.0, num_heads=self.hidden_channels // 16, shortcut=shortcut) for _ in range(n)])

        # Quaternion attention block (ensure gc, ec, nh are defined or pass as init arguments)
        self.attn = MaxSigmoidAttnBlock(self.hidden_channels, self.hidden_channels)

        # Final quaternion convolution to restore channels
        self.cv2 = QConv2d((2 + n) * self.hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(out_channels)
        self.act2 = QReLU()

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the C2PSA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, 4, H, W).
            guide (Optional[torch.Tensor]): Optional guide tensor for attention.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, 4, H, W).
        """
        # Initial conv-bn-activation
        y = self.act1(self.bn1(self.cv1(x)))

        # Split into two branches
        y_a, y_b = y.chunk(2, dim=1)

        # Pass through PSABlock instances
        psablock_out = self.m(y_b)

        # Apply attention
        attn_out = self.attn(psablock_out, guide) if guide is not None else self.attn(psablock_out)

        # Concatenate features
        y = torch.cat([y_a, psablock_out, attn_out], dim=1)

        # Final conv-bn-activation
        y = self.act2(self.bn2(self.cv2(y)))
        
        return y



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

