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
