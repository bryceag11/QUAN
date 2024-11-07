# Quaternion batch normalization
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QBN', 'VQBN', 'IQBN']

class QBN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        """
        Whitening Quaternion Batch Normalization Layer.

        Args:
            num_features (int): Number of quaternion features.
            eps (float): A value added to the denominator for numerical stability.
        """
        super(QBN, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # Initialize Gamma and Beta as learnable parameters (each is a quaternion)
        self.gamma = nn.Parameter(torch.ones(num_features, 4))
        self.beta = nn.Parameter(torch.zeros(num_features, 4))

    def forward(self, x):
        """
        Forward pass for Quaternion Batch Normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, 4, ...).

        Returns:
            torch.Tensor: Normalized and affine-transformed tensor with the same shape as input.
        """
        # Assume x has shape (batch_size, num_features, 4, ...)
        # We'll treat additional dimensions as part of the batch for normalization
        batch_size, num_features, num_quat, *rest = x.shape
        assert num_quat == 4, "The quaternion dimension must be 4."

        # Reshape to (batch_size, num_features, 4, N) where N is the product of other dimensions
        N = 1
        for dim in rest:
            N *= dim
        x = x.view(batch_size, num_features, 4, N)

        # Compute mean E(x) over the batch and spatial dimensions
        E = x.mean(dim=0).mean(dim=-1)  # Shape: (num_features, 4)

        # Center the input
        x_centered = x - E.unsqueeze(-1)  # Shape: (num_features, 4, N)

        # Compute covariance matrix V(x) for each feature
        # V(x) has shape (num_features, 4, 4)
        V = torch.zeros(self.num_features, 4, 4, device=x.device, dtype=x.dtype)
        for f in range(self.num_features):
            # x_centered[f] has shape (4, N)
            # Compute covariance: (x_centered @ x_centered.T) / N
            V[f] = (x_centered[f] @ x_centered[f].transpose(0, 1)) / N
            # Add epsilon to the diagonal for numerical stability
            V[f].diag().add_(self.eps)

        # Invert the covariance matrix
        V_inv = torch.inverse(V)  # Shape: (num_features, 4, 4)

        # Perform Cholesky decomposition on V_inv to get W
        # W is lower triangular such that W @ W.T = V_inv
        try:
            W = torch.linalg.cholesky(V_inv)  # Shape: (num_features, 4, 4)
        except RuntimeError:
            # If V_inv is not positive definite, add epsilon to the diagonal and retry
            V_inv += torch.eye(4, device=x.device).unsqueeze(0) * self.eps
            W = torch.linalg.cholesky(V_inv)

        # Apply whitening: ~x = W @ (x - E)
        # x_centered has shape (num_features, 4, N)
        # W has shape (num_features, 4, 4)
        x_whitened = torch.matmul(W, x_centered)  # Shape: (num_features, 4, N)

        # Reshape back to original shape
        x_whitened = x_whitened.view(batch_size, num_features, 4, *rest)

        # Apply affine transformation: Gamma * ~x + Beta
        # Gamma and Beta have shape (num_features, 4)
        # Need to reshape Gamma and Beta for broadcasting
        gamma = self.gamma.view(1, self.num_features, 4, *([1] * len(rest)))
        beta = self.beta.view(1, self.num_features, 4, *([1] * len(rest)))
        out = gamma * x_whitened + beta

        return out



class VQBN(nn.Module):
    """
    Variance Quaternion Batch Normalization (VQBN).
    
    Normalizes all components of each quaternion jointly using a shared variance.
    
    VQBN produces zero mean and unit variance inputs by applying isotropic scaling, thereby maintaining
    intercomponent relationships. This approach assumes that quaternion components are correlated and should
    be scaled uniformly.
    
    Compared to IndependentQuaternionBatchNorm (ICQBN), VQBN uses a single variance value per quaternion
    (across all components), whereas ICQBN normalizes each component independently.
    
    Advantages of VQBN:
        - Preserves intercomponent relationships by enforcing isotropic scaling.
        - Reduces the number of learnable parameters compared to ICQBN.
    
    Disadvantages of VQBN:
        - Less flexible as it cannot adapt to variations across different quaternion components.
    """
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1):
        """
        Initializes the Variance Quaternion Batch Normalization layer.
        
        Args:
            num_features (int): Number of quaternion features (channels must be multiples of 4).
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): The value used for the running_mean and running_var computation.
        """
        super(VQBN, self).__init__()
        assert num_features %4 ==0, "Number of features must be a multiple of 4 for quaternions."
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters gamma (scale) and beta (shift) for each quaternion
        self.gamma = nn.Parameter(torch.ones(num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_features, 1, 1))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Variance Quaternion Batch Normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if self.training:
            # Compute mean across batch and spatial dimensions for each quaternion
            mean = x.mean(dim=(0, 3, 4), keepdim=True)  # Shape: (1, C, 4, 1, 1)
            # Compute variance across batch and spatial dimensions, assuming isotropic scaling
            var = x.var(dim=(0, 3, 4), unbiased=False, keepdim=True)  # Shape: (1, C, 4, 1, 1)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize with shared variance across quaternion components
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply scale and shift
        out = self.gamma * x_norm + self.beta
        return out


class IQBN(nn.Module):
    """
    Independent Quaternion Batch Normalization (ICQBN).
    
    Normalizes each quaternion component independently to have zero mean and unit variance.
    
    This is different from Variance Quaternion Batch Normalization (VQBN), which normalizes
    all components of a quaternion jointly using a shared variance. ICQBN treats each quaternion component
    as an independent scalar feature, allowing for more flexible normalization but potentially losing
    intercomponent correlations.
    
    VQBN:
        - Uses a shared variance across all quaternion components.
        - Preserves intercomponent relationships by applying isotropic scaling.
        - Computationally efficient as it avoids covariance matrix decomposition.
    
    ICQBN:
        - Normalizes each quaternion component independently.
        - Does not preserve intercomponent relationships.
        - Equally computationally efficient.
    """
    def __init__(self, num_features: int, eps: float=1e-5, momentum: float=0.1):
        """
        Initializes the Independent Quaternion Batch Normalization layer.
        
        Args:
            num_features (int): Number of quaternion features (channels must be multiples of 4).
            eps (float): A value added to the denominator for numerical stability.
            momentum (float): The value used for the running_mean and running_var computation.
        """
        super(IQBN, self).__init__()
        assert num_features %4 ==0, "Number of features must be a multiple of 4 for quaternions."
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters gamma (scale) and beta (shift) for each quaternion component
        self.gamma = nn.Parameter(torch.ones(num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_features, 1, 1))

        # Running statistics (buffered for inference)
        self.register_buffer('running_mean', torch.zeros(num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Independent Quaternion Batch Normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
        
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if self.training:
            # Compute mean and variance across batch and spatial dimensions for each quaternion component
            mean = x.mean(dim=(0, 3, 4), keepdim=True)  # Shape: (1, C, 4, 1, 1)
            var = x.var(dim=(0, 3, 4), unbiased=False, keepdim=True)  # Shape: (1, C, 4, 1, 1)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics for inference
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply scale and shift
        out = self.gamma * x_norm + self.beta
        return out
