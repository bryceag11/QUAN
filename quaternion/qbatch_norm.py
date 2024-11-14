# Quaternion batch normalization
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QBN', 'VQBN', 'IQBN']


# Whitening Quaternion Batch Normalization
class QBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B*Q, C, H, W]
                             where B is batch size, Q is quaternion dimension (4)
        """
        if self.training:
            # Compute mean and var across batch*quaternion and spatial dimensions
            mean = x.mean(dim=(0, 2, 3))  # [C]
            var = x.var(dim=(0, 2, 3), unbiased=False)  # [C]
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

# Variance Quaternion Batch Normalization
class VQBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B*Q, C, H, W]
                             where B is batch size, Q is quaternion dimension (4)
        """
        if self.training:
            # Compute stats
            mean = x.mean(dim=(0, 2, 3))  # [C]
            # Shared variance across quaternion components
            var = x.var(dim=(0, 2, 3), unbiased=False)  # [C]
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize with shared variance
        x_norm = (x - mean[None, :, None, None]) / (var[None, :, None, None] + self.eps).sqrt()
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

class IQBN(nn.Module):
    """Independent Quaternion Batch Normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        """
        Forward pass for IQBN.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B*Q, C, H, W]
                             where B is batch size, Q is quaternion dimension
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        if self.training:
            # Compute stats over batch*quaternion and spatial dimensions
            mean = x.mean(dim=(0, 2, 3))  # [C]
            var = x.var(dim=(0, 2, 3), unbiased=False)  # [C]
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.eps)
        
        # Apply affine transform
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]