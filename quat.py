import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional
import math
from tqdm import tqdm


class QuaternionInit:
    """Quaternion initialization for weights in quaternion layers.
    
    This initializer creates a quaternion-valued weight tensor with modulus
    sampled from a chi distribution with 4 degrees of freedom, and a random
    unit quaternion vector.
    
    Args:
        kernel_size: Size of the convolutional kernel
        input_dim: Input dimension (number of input channels / 4)
        weight_dim: Dimensionality of weight tensor (1D, 2D, 3D)
        nb_filters: Number of filters/output channels
        criterion: Initialization criterion ('he' or 'glorot')
        seed: Random seed
    """
    def __init__(self, 
                 kernel_size, 
                 input_dim,
                 weight_dim, 
                 nb_filters=None, 
                 criterion='he', 
                 seed=None):
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * weight_dim
            
        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 31337 if seed is None else seed
        
    def __call__(self, shape):
        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        else:
            kernel_shape = (self.input_dim, self.kernel_size[-1])
            
        # Compute fan_in and fan_out
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.input_dim * receptive_field_size
        fan_out = self.nb_filters * receptive_field_size
        
        # Set scale based on initialization criterion
        if self.criterion == 'glorot':
            scale = 1.0 / np.sqrt(2 * (fan_in + fan_out))
        elif self.criterion == 'he':
            scale = 1.0 / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        
        # Setup random generators with seed
        rng = np.random.RandomState(self.seed)
        
        # Sample modulus from Chi distribution with 4 degrees of freedom
        flat_size = np.prod(kernel_shape)
        modulus = np.random.rayleigh(scale=scale, size=flat_size)
        modulus = modulus.reshape(kernel_shape)
        
        # Sample phase uniformly
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        
        # Make random unit vector for quaternion vector part
        def make_rand_vector(dims):
            vec = [np.random.normal(0, 1) for i in range(dims)]
            mag = np.sqrt(sum(x**2 for x in vec))
            return [x/mag for x in vec]
        
        # Generate random unit vectors
        u_i = np.zeros(flat_size)
        u_j = np.zeros(flat_size)
        u_k = np.zeros(flat_size)
        
        for u in range(flat_size):
            unit = make_rand_vector(3)
            u_i[u] = unit[0]
            u_j[u] = unit[1]
            u_k[u] = unit[2]
            
        u_i = u_i.reshape(kernel_shape)
        u_j = u_j.reshape(kernel_shape)
        u_k = u_k.reshape(kernel_shape)
        
        # Create the quaternion-valued weight tensor
        weight_r = modulus * np.cos(phase)
        weight_i = modulus * u_i * np.sin(phase)
        weight_j = modulus * u_j * np.sin(phase)
        weight_k = modulus * u_k * np.sin(phase)
        
        return weight_r, weight_i, weight_j, weight_k


class QuaternionDense(nn.Module):
    """
    Quaternion dense (fully connected) layer
    
    Args:
        in_features: Number of quaternion input features (total features / 4)
        out_features: Number of quaternion output features
        bias: Whether to use bias
        init_criterion: Initialization criterion ('he' or 'glorot')
        seed: Random seed for initialization
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 init_criterion: str = 'he',
                 seed: Optional[int] = None):
        
        super(QuaternionDense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_criterion = init_criterion
        self.seed = seed
        
        # Create weight parameters for each part of the quaternion
        self.weight_r = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_i = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_j = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_k = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias term
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using quaternion initialization
        quaternion_init = QuaternionInit(
            kernel_size=(1,),
            input_dim=self.in_features,
            weight_dim=0,
            nb_filters=self.out_features,
            criterion=self.init_criterion,
            seed=self.seed
        )
        
        # Get initialization values
        weight_r, weight_i, weight_j, weight_k = quaternion_init(None)
        
        # Assign values to parameters
        with torch.no_grad():
            self.weight_r.data = torch.FloatTensor(weight_r)
            self.weight_i.data = torch.FloatTensor(weight_i)
            self.weight_j.data = torch.FloatTensor(weight_j)
            self.weight_k.data = torch.FloatTensor(weight_k)
        
        # Initialize bias
        if self.bias is not None:
            self.bias.data.zero_()
    
    def forward(self, x):
        """
        Forward pass of the quaternion dense layer
        
        Args:
            x: Input tensor of shape [batch, 4*in_features]
            
        Returns:
            Output tensor of shape [batch, 4*out_features]
        """
        # Split input into quaternion components
        batch_size = x.size(0)
        x_r = x[:, :self.in_features]
        x_i = x[:, self.in_features:2*self.in_features]
        x_j = x[:, 2*self.in_features:3*self.in_features]
        x_k = x[:, 3*self.in_features:]
        
        # Perform Hamilton product between input and weights
        # r = r*r - i*i - j*j - k*k
        # i = r*i + i*r + j*k - k*j
        # j = r*j - i*k + j*r + k*i
        # k = r*k + i*j - j*i + k*r
        
        # Real part of the result
        out_r = F.linear(x_r, self.weight_r) - \
                F.linear(x_i, self.weight_i) - \
                F.linear(x_j, self.weight_j) - \
                F.linear(x_k, self.weight_k)
        
        # I component of the result
        out_i = F.linear(x_r, self.weight_i) + \
                F.linear(x_i, self.weight_r) + \
                F.linear(x_j, self.weight_k) - \
                F.linear(x_k, self.weight_j)
        
        # J component of the result
        out_j = F.linear(x_r, self.weight_j) - \
                F.linear(x_i, self.weight_k) + \
                F.linear(x_j, self.weight_r) + \
                F.linear(x_k, self.weight_i)
        
        # K component of the result
        out_k = F.linear(x_r, self.weight_k) + \
                F.linear(x_i, self.weight_j) - \
                F.linear(x_j, self.weight_i) + \
                F.linear(x_k, self.weight_r)
        
        # Concatenate output components
        output = torch.cat([out_r, out_i, out_j, out_k], dim=1)
        
        # Add bias if specified
        if self.bias is not None:
            output += self.bias
                
        return output


def get_r(x):
    """Extract the real part from quaternion tensor"""
    dim = x.shape[1] // 4
    return x[:, :dim]


def get_i(x):
    """Extract the i-imaginary part from quaternion tensor"""
    dim = x.shape[1] // 4
    return x[:, dim:2*dim]


def get_j(x):
    """Extract the j-imaginary part from quaternion tensor"""
    dim = x.shape[1] // 4
    return x[:, 2*dim:3*dim]


def get_k(x):
    """Extract the k-imaginary part from quaternion tensor"""
    dim = x.shape[1] // 4
    return x[:, 3*dim:]


class GetR(nn.Module):
    """Layer that extracts the real part from quaternion tensor"""
    def forward(self, x):
        return get_r(x)


class GetI(nn.Module):
    """Layer that extracts the i-imaginary part from quaternion tensor"""
    def forward(self, x):
        return get_i(x)


class GetJ(nn.Module):
    """Layer that extracts the j-imaginary part from quaternion tensor"""
    def forward(self, x):
        return get_j(x)


class GetK(nn.Module):
    """Layer that extracts the k-imaginary part from quaternion tensor"""
    def forward(self, x):
        return get_k(x)


class QuaternionConv(nn.Module):
    """
    Base quaternion convolution layer that performs quaternion convolution operation.
    
    A quaternion convolution applies a quaternion multiplication operation to
    input tensors, where the input is expected to have 4 channels for each 
    quaternion unit (r, i, j, k parts).
    
    Args:
        in_channels: Number of quaternion input channels (total channels / 4)
        out_channels: Number of quaternion output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        bias: Whether to add a learnable bias
        padding_mode: Mode of padding ('zeros', 'reflect', 'replicate', 'circular')
        init_criterion: Initialization criterion ('he' or 'glorot')
        weight_dim: Dimensionality of weight tensor (1D, 2D, 3D)
        seed: Random seed for initialization
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 init_criterion: str = 'he',
                 weight_dim: int = 2,
                 seed: Optional[int] = None):
        
        super(QuaternionConv, self).__init__()
        
        self.in_channels = in_channels  # Input quaternion units
        self.out_channels = out_channels  # Output quaternion units
        
        # Handle different kernel size formats
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * weight_dim
        self.kernel_size = kernel_size
        
        # Handle different stride formats
        if isinstance(stride, int):
            stride = (stride,) * weight_dim
        self.stride = stride
        
        # Handle different padding formats
        if isinstance(padding, int):
            padding = (padding,) * weight_dim
        self.padding = padding
        
        # Handle different dilation formats
        if isinstance(dilation, int):
            dilation = (dilation,) * weight_dim
        self.dilation = dilation
        
        self.padding_mode = padding_mode
        self.init_criterion = init_criterion
        self.weight_dim = weight_dim
        self.seed = seed
        
        # Create weight Parameters for each part of the quaternion
        if weight_dim == 1:
            self.weight_r = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_i = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_j = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_k = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        elif weight_dim == 2:
            self.weight_r = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_i = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_j = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_k = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        elif weight_dim == 3:
            self.weight_r = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_i = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_j = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            self.weight_k = nn.Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        
        # Bias term (applied to all 4 components of quaternion)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize weights using quaternion initialization
        quaternion_init = QuaternionInit(
            kernel_size=self.kernel_size,
            input_dim=self.in_channels,
            weight_dim=self.weight_dim,
            nb_filters=self.out_channels,
            criterion=self.init_criterion,
            seed=self.seed
        )
        
        # Get initialization values
        weight_r, weight_i, weight_j, weight_k = quaternion_init(None)
        
        # Initialize the weights with values from quaternion initializer
        # First permute from the initializer format to PyTorch format
        if self.weight_dim == 1:
            weight_r = np.transpose(weight_r, (2, 0, 1))
            weight_i = np.transpose(weight_i, (2, 0, 1))
            weight_j = np.transpose(weight_j, (2, 0, 1))
            weight_k = np.transpose(weight_k, (2, 0, 1))
        elif self.weight_dim == 2:
            weight_r = np.transpose(weight_r, (3, 2, 0, 1))
            weight_i = np.transpose(weight_i, (3, 2, 0, 1))
            weight_j = np.transpose(weight_j, (3, 2, 0, 1))
            weight_k = np.transpose(weight_k, (3, 2, 0, 1))
        elif self.weight_dim == 3:
            weight_r = np.transpose(weight_r, (4, 3, 0, 1, 2))
            weight_i = np.transpose(weight_i, (4, 3, 0, 1, 2))
            weight_j = np.transpose(weight_j, (4, 3, 0, 1, 2))
            weight_k = np.transpose(weight_k, (4, 3, 0, 1, 2))
            
        # Assign values to parameters
        with torch.no_grad():
            self.weight_r.data = torch.FloatTensor(weight_r)
            self.weight_i.data = torch.FloatTensor(weight_i)
            self.weight_j.data = torch.FloatTensor(weight_j)
            self.weight_k.data = torch.FloatTensor(weight_k)
        
        # Initialize bias
        if self.bias is not None:
            self.bias.data.zero_()
    
    def quaternion_conv(self, x):
        """
        Performs a quaternion convolution operation.
        
        Args:
            x: Input tensor of shape [batch, 4*in_channels, *spatial_dims]
            
        Returns:
            Output tensor of shape [batch, 4*out_channels, *spatial_dims]
        """
        # Split input into quaternion components
        batch_size = x.size(0)
        x_r, x_i, x_j, x_k = self.split_quaternion(x)
        
        # Perform Hamilton product between input and weights
        # r = r*r - i*i - j*j - k*k
        # i = r*i + i*r + j*k - k*j
        # j = r*j - i*k + j*r + k*i
        # k = r*k + i*j - j*i + k*r
        
        if self.weight_dim == 1:
            conv_op = F.conv1d
        elif self.weight_dim == 2:
            conv_op = F.conv2d
        elif self.weight_dim == 3:
            conv_op = F.conv3d
        else:
            raise ValueError(f"Unsupported weight_dim: {self.weight_dim}")
        
        # Convolution operations for Hamilton product components
        out_r = conv_op(x_r, self.weight_r, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_i, self.weight_i, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_j, self.weight_j, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_k, self.weight_k, None, self.stride, self.padding, self.dilation)
                
        out_i = conv_op(x_r, self.weight_i, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_i, self.weight_r, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_j, self.weight_k, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_k, self.weight_j, None, self.stride, self.padding, self.dilation)
                
        out_j = conv_op(x_r, self.weight_j, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_i, self.weight_k, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_j, self.weight_r, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_k, self.weight_i, None, self.stride, self.padding, self.dilation)
                
        out_k = conv_op(x_r, self.weight_k, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_i, self.weight_j, None, self.stride, self.padding, self.dilation) - \
                conv_op(x_j, self.weight_i, None, self.stride, self.padding, self.dilation) + \
                conv_op(x_k, self.weight_r, None, self.stride, self.padding, self.dilation)
        
        # Concatenate output components
        return torch.cat([out_r, out_i, out_j, out_k], dim=1)
    
    def split_quaternion(self, x):
        """Split a quaternion tensor into its four components"""
        batch_size = x.size(0)
        dim = x.size(1) // 4
        x_r = x[:, :dim]
        x_i = x[:, dim:2*dim]
        x_j = x[:, 2*dim:3*dim]
        x_k = x[:, 3*dim:]
        return x_r, x_i, x_j, x_k
    
    def forward(self, x):
        """Forward pass of the quaternion convolution"""
        output = self.quaternion_conv(x)
        
        # Add bias if specified
        if self.bias is not None:
            if self.weight_dim == 1:
                output += self.bias.view(1, -1, 1)
            elif self.weight_dim == 2:
                output += self.bias.view(1, -1, 1, 1)
            elif self.weight_dim == 3:
                output += self.bias.view(1, -1, 1, 1, 1)
                
        return output


class QuaternionConv2D(QuaternionConv):
    """
    2D quaternion convolution layer.
    
    This layer creates a quaternion convolution kernel that is convolved with
    the quaternion input to produce a quaternion output tensor.
    
    Args:
        in_channels: Number of quaternion input channels (total channels / 4)
        out_channels: Number of quaternion output channels
        kernel_size: Size of the convolutional kernel
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        bias: Whether to add a learnable bias
        padding_mode: Mode of padding ('zeros', 'reflect', 'replicate', 'circular')
        init_criterion: Initialization criterion ('he' or 'glorot')
        seed: Random seed for initialization
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 init_criterion: str = 'he',
                 seed: Optional[int] = None):
        
        super(QuaternionConv2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            init_criterion=init_criterion,
            weight_dim=2,
            seed=seed
        )


class QuaternionBatchNorm(nn.Module):
    """
    Quaternion Batch Normalization layer.
    
    This layer normalizes a quaternion input tensor by calculating the mean and
    variance of each of the four quaternion components, and using these statistics
    to normalize the input. It also maintains a running mean and variance for use
    during evaluation.
    
    Args:
        num_features: Number of quaternion features (total channels / 4)
        eps: Small constant for numerical stability
        momentum: Momentum factor for running statistics
        affine: Whether to apply learnable affine parameters
        track_running_stats: Whether to track running statistics
    """
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        
        super(QuaternionBatchNorm, self).__init__()
        
        self.num_features = num_features  # Number of quaternion features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Total number of channels is 4 * num_features
        total_features = 4 * num_features
        
        # Register running statistics buffers
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(total_features))
            self.register_buffer('running_var', torch.ones(total_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        
        # Register affine parameters
        if self.affine:
            # Parameters for the 4x4 transformation matrix for each quaternion unit
            # Diagonal terms
            self.gamma_rr = nn.Parameter(torch.ones(num_features) / math.sqrt(4))
            self.gamma_ii = nn.Parameter(torch.ones(num_features) / math.sqrt(4))
            self.gamma_jj = nn.Parameter(torch.ones(num_features) / math.sqrt(4))
            self.gamma_kk = nn.Parameter(torch.ones(num_features) / math.sqrt(4))
            
            # Off-diagonal terms
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.gamma_rj = nn.Parameter(torch.zeros(num_features))
            self.gamma_rk = nn.Parameter(torch.zeros(num_features))
            self.gamma_ij = nn.Parameter(torch.zeros(num_features))
            self.gamma_ik = nn.Parameter(torch.zeros(num_features))
            self.gamma_jk = nn.Parameter(torch.zeros(num_features))
            
            # Bias terms
            self.beta = nn.Parameter(torch.zeros(total_features))
        else:
            self.register_parameter('gamma_rr', None)
            self.register_parameter('gamma_ii', None)
            self.register_parameter('gamma_jj', None)
            self.register_parameter('gamma_kk', None)
            self.register_parameter('gamma_ri', None)
            self.register_parameter('gamma_rj', None)
            self.register_parameter('gamma_rk', None)
            self.register_parameter('gamma_ij', None)
            self.register_parameter('gamma_ik', None)
            self.register_parameter('gamma_jk', None)
            self.register_parameter('beta', None)
    
    def _calculate_quaternion_stats(self, x):
        """Calculate quaternion batch statistics"""
        batch_size = x.size(0)
        x_r, x_i, x_j, x_k = self.split_quaternion(x)
        
        # Calculate means for each component
        mean_r = x_r.mean([0] + list(range(2, x_r.dim())))
        mean_i = x_i.mean([0] + list(range(2, x_i.dim())))
        mean_j = x_j.mean([0] + list(range(2, x_j.dim())))
        mean_k = x_k.mean([0] + list(range(2, x_k.dim())))
        
        # Combine means
        mean = torch.cat([mean_r, mean_i, mean_j, mean_k], dim=0)
        
        # Center inputs
        centered_r = x_r - mean_r.view(1, -1, *([1] * (x_r.dim() - 2)))
        centered_i = x_i - mean_i.view(1, -1, *([1] * (x_i.dim() - 2)))
        centered_j = x_j - mean_j.view(1, -1, *([1] * (x_j.dim() - 2)))
        centered_k = x_k - mean_k.view(1, -1, *([1] * (x_k.dim() - 2)))
        
        # Calculate variances and covariances
        var_r = (centered_r ** 2).mean([0] + list(range(2, centered_r.dim())))
        var_i = (centered_i ** 2).mean([0] + list(range(2, centered_i.dim())))
        var_j = (centered_j ** 2).mean([0] + list(range(2, centered_j.dim())))
        var_k = (centered_k ** 2).mean([0] + list(range(2, centered_k.dim())))
        
        # Covariances between components
        cov_ri = (centered_r * centered_i).mean([0] + list(range(2, centered_r.dim())))
        cov_rj = (centered_r * centered_j).mean([0] + list(range(2, centered_r.dim())))
        cov_rk = (centered_r * centered_k).mean([0] + list(range(2, centered_r.dim())))
        cov_ij = (centered_i * centered_j).mean([0] + list(range(2, centered_i.dim())))
        cov_ik = (centered_i * centered_k).mean([0] + list(range(2, centered_i.dim())))
        cov_jk = (centered_j * centered_k).mean([0] + list(range(2, centered_j.dim())))
        
        # Combine variances
        var = torch.cat([var_r, var_i, var_j, var_k], dim=0)
        
        return mean, var, centered_r, centered_i, centered_j, centered_k, \
               var_r, var_i, var_j, var_k, cov_ri, cov_rj, cov_rk, cov_ij, cov_ik, cov_jk
    
    def _quaternion_standardization(self, centered_r, centered_i, centered_j, centered_k,
                                   var_r, var_i, var_j, var_k, 
                                   cov_ri, cov_rj, cov_rk, cov_ij, cov_ik, cov_jk):
        """Standardize quaternion input"""
        # Add epsilon for numerical stability
        var_r = var_r + self.eps
        var_i = var_i + self.eps
        var_j = var_j + self.eps
        var_k = var_k + self.eps
        
        # Cholesky decomposition of 4x4 covariance matrix
        w_rr = torch.sqrt(var_r)
        w_ri = cov_ri / w_rr
        w_ii = torch.sqrt(var_i - w_ri * w_ri)
        w_rj = cov_rj / w_rr
        w_ij = (cov_ij - w_ri * w_rj) / w_ii
        w_jj = torch.sqrt(var_j - (w_ij * w_ij + w_rj * w_rj))
        w_rk = cov_rk / w_rr
        w_ik = (cov_ik - w_ri * w_rk) / w_ii
        w_jk = (cov_jk - (w_ij * w_ik + w_rj * w_rk)) / w_jj
        w_kk = torch.sqrt(var_k - (w_jk * w_jk + w_ik * w_ik + w_rk * w_rk))
        
        # Reshape for broadcasting
        ndim = centered_r.dim()
        broadcast_shape = [1, -1] + [1] * (ndim - 2)
        
        # Reshape weight components for broadcasting
        w_rr = w_rr.view(*broadcast_shape)
        w_ri = w_ri.view(*broadcast_shape)
        w_rj = w_rj.view(*broadcast_shape)
        w_rk = w_rk.view(*broadcast_shape)
        w_ii = w_ii.view(*broadcast_shape)
        w_ij = w_ij.view(*broadcast_shape)
        w_ik = w_ik.view(*broadcast_shape)
        w_jj = w_jj.view(*broadcast_shape)
        w_jk = w_jk.view(*broadcast_shape)
        w_kk = w_kk.view(*broadcast_shape)
        
        # Apply normalization: multiply by inverse of Cholesky factor
        # Equivalent to solving triangular system for each component
        r_out = (w_rr * centered_r + w_ri * centered_i + w_rj * centered_j + w_rk * centered_k)
        i_out = (w_ii * centered_i + w_ij * centered_j + w_ik * centered_k)
        j_out = (w_jj * centered_j + w_jk * centered_k)
        k_out = (w_kk * centered_k)
        
        return r_out, i_out, j_out, k_out

    def _apply_affine_transform(self, r_normed, i_normed, j_normed, k_normed):
        """Apply learnable affine transformation"""
        ndim = r_normed.dim()
        broadcast_shape = [1, -1] + [1] * (ndim - 2)
        
        # Reshape gamma and beta parameters for broadcasting
        gamma_rr = self.gamma_rr.view(*broadcast_shape)
        gamma_ri = self.gamma_ri.view(*broadcast_shape)
        gamma_rj = self.gamma_rj.view(*broadcast_shape)
        gamma_rk = self.gamma_rk.view(*broadcast_shape)
        gamma_ii = self.gamma_ii.view(*broadcast_shape)
        gamma_ij = self.gamma_ij.view(*broadcast_shape)
        gamma_ik = self.gamma_ik.view(*broadcast_shape)
        gamma_jj = self.gamma_jj.view(*broadcast_shape)
        gamma_jk = self.gamma_jk.view(*broadcast_shape)
        gamma_kk = self.gamma_kk.view(*broadcast_shape)
        
        # Apply transformation matrix
        r_out = gamma_rr * r_normed + gamma_ri * i_normed + gamma_rj * j_normed + gamma_rk * k_normed
        i_out = gamma_ri * r_normed + gamma_ii * i_normed + gamma_ij * j_normed + gamma_ik * k_normed
        j_out = gamma_rj * r_normed + gamma_ij * i_normed + gamma_jj * j_normed + gamma_jk * k_normed
        k_out = gamma_rk * r_normed + gamma_ik * i_normed + gamma_jk * j_normed + gamma_kk * k_normed
        
        # Combine outputs
        output = torch.cat([r_out, i_out, j_out, k_out], dim=1)
        
        # Add bias if using affine parameters
        if self.beta is not None:
            beta_shape = [1, -1] + [1] * (output.dim() - 2)
            output = output + self.beta.view(*beta_shape)
        
        return output

    def forward(self, x):
        """
        Forward pass of the quaternion batch normalization layer.
        
        Args:
            x: Input tensor of shape [batch, 4*channels, *spatial_dims]
                
        Returns:
            Normalized tensor of the same shape
        """
        # Handle different training modes
        if self.training:
            # Calculate batch statistics
            mean, var, centered_r, centered_i, centered_j, centered_k, \
            var_r, var_i, var_j, var_k, cov_ri, cov_rj, cov_rk, cov_ij, cov_ik, cov_jk = \
            self._calculate_quaternion_stats(x)
            
            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
                    self.num_batches_tracked += 1
            
            # Standardize the input
            r_normed, i_normed, j_normed, k_normed = self._quaternion_standardization(
                centered_r, centered_i, centered_j, centered_k,
                var_r, var_i, var_j, var_k,
                cov_ri, cov_rj, cov_rk, cov_ij, cov_ik, cov_jk
            )
        else:
            # Use running statistics in evaluation mode
            if not self.track_running_stats:
                raise ValueError("Cannot use batch normalization in eval mode without tracking statistics")
            
            # Center the input using running mean
            batch_size = x.size(0)
            ndim = x.dim()
            mean_shape = [1, -1] + [1] * (ndim - 2)
            running_mean = self.running_mean.view(*mean_shape)
            
            x_centered = x - running_mean
            
            # Split into quaternion components
            x_r, x_i, x_j, x_k = self.split_quaternion(x_centered)
            
            # Use pre-computed whitening from running statistics
            # For simplicity, we'll use a basic normalization with running variance in eval mode
            # This is a simplification of the full quaternion standardization
            var_shape = [1, -1] + [1] * (ndim - 2)
            running_var = self.running_var.view(*var_shape)
            
            # Normalize each component separately (simplified approach)
            dim = x.size(1) // 4
            r_normed = x_r / (torch.sqrt(running_var[:, :dim] + self.eps))
            i_normed = x_i / (torch.sqrt(running_var[:, dim:2*dim] + self.eps))
            j_normed = x_j / (torch.sqrt(running_var[:, 2*dim:3*dim] + self.eps))
            k_normed = x_k / (torch.sqrt(running_var[:, 3*dim:] + self.eps))
        
        # Apply affine transformation if specified
        if self.affine:
            output = self._apply_affine_transform(r_normed, i_normed, j_normed, k_normed)
        else:
            output = torch.cat([r_normed, i_normed, j_normed, k_normed], dim=1)
        
        return output
    
    def split_quaternion(self, x):
        """Split a quaternion tensor into its four components"""
        batch_size = x.size(0)
        dim = x.size(1) // 4
        x_r = x[:, :dim]
        x_i = x[:, dim:2*dim]
        x_j = x[:, 2*dim:3*dim]
        x_k = x[:, 3*dim:]
        return x_r, x_i, x_j, x_k