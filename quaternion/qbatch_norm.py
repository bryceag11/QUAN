# Quaternion batch normalization
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QBN', 'VQBN', 'IQBN']


# Whitening Quaternion Batch Normalization

# class QBN(nn.Module):
#     """
#     Quaternion-Aware Whitening Batch Normalization
#     Performs whitening considering quaternion interrelationships
#     """
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
        
#         # Single set of parameters for whole quaternion
#         self.gamma = nn.Parameter(torch.ones(num_features))
#         self.beta = nn.Parameter(torch.zeros(num_features))
        
#         # Running stats for quaternion as whole unit
#         self.register_buffer('running_mean', torch.zeros(num_features, 4))  # [C, Q]
#         self.register_buffer('running_cov', torch.eye(4).repeat(num_features, 1, 1))  # [C, Q, Q]
    
#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): Input tensor [B, C, Q, H, W]
#         """
#         B, C, Q, H, W = x.shape
#         assert Q == 4, "Expected quaternion input with 4 components"
        
#         # Reshape to [B*H*W, C, Q]
#         x_reshaped = x.reshape(B, C, Q, H*W)
        
#         if self.training:
#             # Compute mean across batch and spatial dimensions
#             mean = x_reshaped.mean(dim=(0, 3))  # [C, Q]
            
#             # Center the data
#             x_centered = x_reshaped - mean[None, :, :, None]
            
#             # Compute covariance matrix for each channel
#             # Reshape to [B*C, Q, H*W]
#             x_centered_flat = x_centered.permute(1, 0, 2, 3).reshape(C, B, Q, H*W)
            
#             # Compute covariance for each channel
#             cov_list = []
#             for c in range(C):
#                 channel_x_centered = x_centered[:, c].permute(1, 0, 2)

#                 # [B, Q, H*W] -> [Q, Q]
#                 channel_cov = torch.matmul(channel_x_centered, channel_x_centered.transpose(1, 2)) / (B * H * W)
#                 cov_list.append(channel_cov)
            
#             cov = torch.stack(cov_list)  # [C, Q, Q]
            
#             # Update running stats
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
#             self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * cov
#         else:
#             mean = self.running_mean
#             cov = self.running_cov
        
#         # Compute whitening matrices using eigendecomposition
#         whitening_matrices = []
#         for c in range(C):
#             eigenvalues, eigenvectors = torch.linalg.eigh(cov[c])
#             # Add eps to eigenvalues for numerical stability
#             whitening = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues + self.eps)) @ eigenvectors.t()
#             whitening_matrices.append(whitening)
#         whitening_matrices = torch.stack(whitening_matrices)  # [C, Q, Q]
        
#         # Center the data
#         x_centered = x_reshaped - mean[None, :, :, None]
        
#         # Apply whitening transformation
#         # Use einsum to apply channel-wise whitening
#         x_whitened = torch.einsum('bcqn,cqk->bckn', x_centered, whitening_matrices)
        
#         # Reshape back to original shape and apply learnable parameters
#         x_out = x_whitened.reshape(B, C, Q, H, W)
#         return self.gamma[None, :, None, None, None] * x_out + self.beta[None, :, None, None, None]

# class QBN(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.num_features = num_features // 4
#         self.eps = eps
#         self.momentum = momentum
        
#         self.register_buffer('running_mean', torch.zeros(self.num_features, 4))
#         self.register_buffer('running_cov', torch.eye(4).unsqueeze(0).repeat(self.num_features, 1, 1))
        
#         self.gamma = nn.Parameter(torch.ones(self.num_features, 4))
#         self.beta = nn.Parameter(torch.zeros(self.num_features, 4))

#     def forward(self, x):
#         B, C, Q, H, W = x.shape
#         assert Q == 4
#         assert C % 4 == 0
#         C_quat = C
        
#         # Reshape to [B*H*W, C_quat, Q]
#         x_flat = x.permute(0, 2, 1, 3, 4).reshape(-1, C_quat, 4)
        
#         if self.training:
#             # Detach for running stats computation to avoid graph retention
#             x_compute = x_flat.detach()
            
#             # Compute mean
#             mean = x_compute.mean(dim=0)
            
#             # Compute covariance
#             x_centered_compute = x_compute - mean.unsqueeze(0)
#             x_centered_compute = x_centered_compute.transpose(0, 1)
            
#             cov = torch.zeros(C_quat, 4, 4, device=x.device)
#             for c in range(C_quat):
#                 feat_centered = x_centered_compute[c]
#                 feat_cov = torch.matmul(feat_centered.t(), feat_centered) / (B * H * W)
#                 cov[c] = feat_cov
            
#             # Update running stats
#             with torch.no_grad():
#                 self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
#                 self.running_cov = (1 - self.momentum) * self.running_cov + self.momentum * cov
        
#         # Use running stats and create differentiable path
#         x_centered = x_flat - self.running_mean.unsqueeze(0)
        
#         # Pre-compute inverse matrices to avoid creating multiple graphs
#         L_invs = []
#         for c in range(C_quat):
#             cov_matrix = self.running_cov[c] + self.eps * torch.eye(4, device=x.device)
#             L = torch.linalg.cholesky(cov_matrix)
#             L_inv = torch.inverse(L)
#             L_invs.append(L_inv)
        
#         # Apply whitening with pre-computed matrices
#         x_whitened = torch.zeros_like(x_flat)
#         for c in range(C_quat):
#             x_whitened[:, c, :] = torch.matmul(x_centered[:, c, :], L_invs[c])
        
#         # Apply scaling and shift
#         x_out = x_whitened * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)
        
#         # Reshape back
#         x_out = x_out.view(B, H, W, C_quat, 4).permute(0, 3, 4, 1, 2)
        
#         return x_out


class VQBN(nn.Module):
    """
    Quaternion-Aware Variance-Only Batch Normalization
    Uses shared variance across quaternion components while maintaining quaternion structure
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Separate scale parameters for each component
        self.gamma_r = nn.Parameter(torch.ones(num_features))
        self.gamma_i = nn.Parameter(torch.ones(num_features))
        self.gamma_j = nn.Parameter(torch.ones(num_features))
        self.gamma_k = nn.Parameter(torch.ones(num_features))
        
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))
        self.beta_j = nn.Parameter(torch.zeros(num_features))
        self.beta_k = nn.Parameter(torch.zeros(num_features))
        
        # Running stats
        self.register_buffer('running_mean_r', torch.zeros(num_features))
        self.register_buffer('running_mean_i', torch.zeros(num_features))
        self.register_buffer('running_mean_j', torch.zeros(num_features))
        self.register_buffer('running_mean_k', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))  # Shared variance
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C, Q, H, W]
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input with 4 components"
        
        # Split into components
        x_r, x_i, x_j, x_k = x.split(1, dim=2)
        
        if self.training:
            # Compute means for each component
            mean_r = x_r.mean(dim=(0, 2, 3, 4))
            mean_i = x_i.mean(dim=(0, 2, 3, 4))
            mean_j = x_j.mean(dim=(0, 2, 3, 4))
            mean_k = x_k.mean(dim=(0, 2, 3, 4))
            
            # Compute shared variance across all components
            x_centered = torch.cat([
                (x_r.squeeze(2) - mean_r[None, :, None, None]),
                (x_i.squeeze(2) - mean_i[None, :, None, None]),
                (x_j.squeeze(2) - mean_j[None, :, None, None]),
                (x_k.squeeze(2) - mean_k[None, :, None, None])
            ], dim=0)
            var = x_centered.var(dim=(0, 2, 3), unbiased=False)  # Shared variance
            
            # Update running stats
            self.running_mean_r = (1 - self.momentum) * self.running_mean_r + self.momentum * mean_r
            self.running_mean_i = (1 - self.momentum) * self.running_mean_i + self.momentum * mean_i
            self.running_mean_j = (1 - self.momentum) * self.running_mean_j + self.momentum * mean_j
            self.running_mean_k = (1 - self.momentum) * self.running_mean_k + self.momentum * mean_k
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean_r, mean_i, mean_j, mean_k = (self.running_mean_r, self.running_mean_i,
                                            self.running_mean_j, self.running_mean_k)
            var = self.running_var
        
        # Normalize using shared variance but separate means
        x_r_norm = ((x_r.squeeze(2) - mean_r[None, :, None, None]) / 
                   (var[None, :, None, None] + self.eps).sqrt())
        x_i_norm = ((x_i.squeeze(2) - mean_i[None, :, None, None]) / 
                   (var[None, :, None, None] + self.eps).sqrt())
        x_j_norm = ((x_j.squeeze(2) - mean_j[None, :, None, None]) / 
                   (var[None, :, None, None] + self.eps).sqrt())
        x_k_norm = ((x_k.squeeze(2) - mean_k[None, :, None, None]) / 
                   (var[None, :, None, None] + self.eps).sqrt())
        
        # Apply learnable parameters while maintaining quaternion structure
        x_r_out = self.gamma_r[None, :, None, None] * x_r_norm + self.beta_r[None, :, None, None]
        x_i_out = self.gamma_i[None, :, None, None] * x_i_norm + self.beta_i[None, :, None, None]
        x_j_out = self.gamma_j[None, :, None, None] * x_j_norm + self.beta_j[None, :, None, None]
        x_k_out = self.gamma_k[None, :, None, None] * x_k_norm + self.beta_k[None, :, None, None]
        
        # Stack back to quaternion format
        return torch.stack([x_r_out, x_i_out, x_j_out, x_k_out], dim=2)


class QBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # One scalar stretch parameter
        self.gamma = nn.Parameter(torch.ones(1))
        # Pure imaginary shift parameter (i,j,k components)
        self.beta = nn.Parameter(torch.zeros(3))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C, Q, H, W]
                            Q=4 quaternion components (real, i, j, k)
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input with 4 components"
        
        # Calculate mean and variance for all components
        mean = x.mean(dim=(0, 1, 3, 4), keepdim=True)  # [1, 1, Q, 1, 1]
        var = x.var(dim=(0, 1, 3, 4), keepdim=True, unbiased=False)  # [1, 1, Q, 1, 1]
        
        # Normalize all components
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply stretch to all components
        x_stretched = self.gamma * x_norm
        
        # Apply shift only to imaginary components
        x_out = x_stretched.clone()
        x_out[:, :, 1:] = x_out[:, :, 1:] + self.beta.view(1, 1, 3, 1, 1)
        
        return x_out

class IQBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # Adjust features for quaternion structure
        self.num_features = num_features // 4
        self.eps = eps
        self.momentum = momentum

        # Create parameters
        self.gamma = nn.Parameter(torch.ones(self.num_features, 4))
        self.beta = nn.Parameter(torch.zeros(self.num_features, 4))
        
        # Register buffers with correct shapes
        self.register_buffer('running_mean', torch.zeros(self.num_features, 4))
        self.register_buffer('running_var', torch.ones(self.num_features, 4))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input with 4 components"
        assert C == self.num_features, f"Expected {self.num_features} quaternion channels, got {C}"

        if not self.training:
            # Use running stats during eval (more efficient)
            mean = self.running_mean.view(1, self.num_features, 4, 1, 1)
            var = self.running_var.view(1, self.num_features, 4, 1, 1)
            x = (x - mean) / (var + self.eps).sqrt()
            return x * self.gamma.view(1, self.num_features, 4, 1, 1) + self.beta.view(1, self.num_features, 4, 1, 1)

        # Training mode
        x_reshaped = x.transpose(1, 2).reshape(B, Q, C, -1)
        
        # Compute batch statistics
        mean = x_reshaped.mean(dim=(0, -1))  # [Q, C]
        var = x_reshaped.var(dim=(0, -1), unbiased=False)  # [Q, C]
        
        # Update running stats
        with torch.no_grad():
            self.running_mean = self.running_mean * (1 - self.momentum) + mean.t() * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var.t() * self.momentum
            self.num_batches_tracked += 1
        
        # Normalize and apply parameters
        mean = mean.view(1, Q, C, 1)
        var = var.view(1, Q, C, 1)
        x_norm = (x_reshaped - mean) / (var + self.eps).sqrt()
        x_norm = x_norm * self.gamma.t().view(1, Q, C, 1) + self.beta.t().view(1, Q, C, 1)
        
        return x_norm.reshape(B, Q, C, H, W).transpose(1, 2)


#     """
#     Quaternion Batch Normalization with careful running stats management
#     """
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum

#         # Separate parameters for each quaternion component
#         self.running_mean = None
#         self.running_var = None

#         # Parameters will now be created dynamically
#         self.gamma = None
#         self.beta = None
#     def forward(self, x):
#         B, C, Q, H, W = x.shape
#         assert Q == 4, "Expected quaternion input with 4 components"

#         # Dynamically create parameters if not already created
#         if self.gamma is None:
#             self.gamma = nn.Parameter(torch.ones(C, 4).to(x.device))
#             self.beta = nn.Parameter(torch.zeros(C, 4).to(x.device))

#             # Initialize running stats
#             self.running_mean = torch.zeros(C, 4).to(x.device)
#             self.running_var = torch.ones(C, 4).to(x.device)

#         # Reshape to [B, C, Q, HW] for easier stats computation
#         x_reshaped = x.reshape(B, C, Q, H*W)

#         if self.training:
#             # Compute mean across batch and spatial dimensions
#             # Mean shape: [C, Q]
#             mean = x_reshaped.mean(dim=(0, 3))

#             # Compute variance across batch and spatial dimensions
#             # Var shape: [C, Q]
#             var = x_reshaped.var(dim=(0, 3))

#             # Update running stats
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
#         else:
#             mean = self.running_mean
#             var = self.running_var

#         # Normalize each quaternion component
#         # Broadcast mean and var to [B, C, Q, H*W]
#         x_norm = (x_reshaped - mean[None, :, :, None]) / torch.sqrt(var[None, :, :, None] + self.eps)

#         # Reshape back to original shape
#         x_scaled = x_norm.reshape(B, C, Q, H, W)

#         # Apply learnable parameters
#         # Ensure correct broadcasting of gamma and beta
#         gamma = self.gamma.view(1, C, 4, 1, 1)
#         beta = self.beta.view(1, C, 4, 1, 1)

#         x_out = gamma * x_scaled + beta

#         return x_out
    
    
