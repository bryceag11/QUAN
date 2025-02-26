#!/usr/bin/env python

import torch
import numpy as np
from torch import Tensor
from typing import Dict, Optional, Union, Tuple, List

class QInit:
    """
    Quaternion weight initialization for PyTorch quaternion neural networks.
    
    Handles initialization of both phase and modulus components for quaternion operations.
    Can be used with both convolutional and dense layers.
    """
    
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, ...]],
                 input_dim: int,
                 weight_dim: int,
                 nb_filters: Optional[int] = None,
                 criterion: str = 'he') -> None:
        """
        Initialize the quaternion weight initializer.
        
        Args:
            kernel_size: Size of convolution kernel
            input_dim: Number of input channels/dimensions
            weight_dim: Dimensionality of weights (0,1,2,3)
            nb_filters: Number of output filters (optional)
            criterion: Weight initialization criterion ('he' or 'glorot')
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.nb_filters = nb_filters
        self.criterion = criterion

    def initialize(self, shape: Tuple[int, ...], device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """
        Generate initialized quaternion weights.
        
        Args:
            shape: Required shape of weights
            device: Device to place weights on
            
        Returns:
            Dictionary containing phase and modulus weights
        """
        if self.nb_filters is not None:
            kernel_shape = self.kernel_size + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        # Calculate fan_in and fan_out for initialization scaling
        if len(kernel_shape) > 2:
            fan_in = kernel_shape[1] * np.prod(kernel_shape[2:])
            fan_out = kernel_shape[0] * np.prod(kernel_shape[2:])
        else:
            fan_in = kernel_shape[1]
            fan_out = kernel_shape[0]

        # Determine initialization scaling factor
        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)

        # Initialize modulus weights
        modulus = torch.empty(kernel_shape, device=device)
        bound = np.sqrt(s) * np.sqrt(3)
        modulus = torch.nn.init.uniform_(modulus, -bound, bound)

        # Initialize phase weights
        phase = torch.empty(kernel_shape, device=device)
        phase = torch.nn.init.uniform_(phase, -np.pi/2, np.pi/2)

        return {
            'modulus': modulus,
            'phase': phase
        }

    @staticmethod
    def get_kernel_size(weight_shape: Tuple[int, ...], dim: int) -> Tuple[int, ...]:
        """
        Extract kernel size from weight shape based on dimensionality.
        
        Args:
            weight_shape: Shape of weights
            dim: Number of dimensions (1,2,3)
            
        Returns:
            Tuple containing kernel size
        """
        if dim == 1:
            return (weight_shape[2],)
        elif dim == 2:
            return (weight_shape[2], weight_shape[3])
        elif dim == 3:
            return (weight_shape[2], weight_shape[3], weight_shape[4])
        else:
            raise ValueError(f"Unsupported number of dimensions: {dim}")
        


import torch
import numpy as np
from typing import Union, Tuple, List, Optional

class QuaternionInit:
    """
    Quaternion weight initialization for PyTorch quaternion neural networks.
    
    This initializer creates quaternion-valued weights with modulus sampled from
    a Chi distribution with 4 degrees of freedom and a random unit quaternion vector.
    
    Args:
        kernel_size: Size of the convolutional kernel
        in_features: Number of input features (divided by 4 for quaternion representation)
        out_features: Number of output features (divided by 4 for quaternion representation)
        criterion: Weight initialization criterion ('he' or 'glorot')
        rng: Random number generator or seed
    """
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, ...]],
        in_features: int,
        out_features: int,
        criterion: str = 'he',
        rng: Optional[Union[np.random.RandomState, int]] = None
    ):
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,)
        else:
            self.kernel_size = kernel_size
            
        self.in_features = in_features
        self.out_features = out_features
        self.criterion = criterion
        
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng or 1234)
    
    def initialize(self, shape=None, device=None):
        """
        Generate quaternion weights with proper initialization.
        
        Returns:
            Tuple of (W_r, W_i, W_j, W_k) for the quaternion components
        """
        # Calculate kernel parameter for quaternion weight init
        if self.criterion == 'glorot':
            # Glorot/Xavier initialization
            fan_in = self.in_features * np.prod(self.kernel_size)
            fan_out = self.out_features * np.prod(self.kernel_size)
            scale = 1.0 / np.sqrt(2.0 * (fan_in + fan_out))
        elif self.criterion == 'he':
            # He initialization (better for ReLU)
            fan_in = self.in_features * np.prod(self.kernel_size)
            scale = 1.0 / np.sqrt(2.0 * fan_in)
        else:
            raise ValueError(f"Unknown initialization criterion: {self.criterion}")
        
        # Get the final weight shape
        if shape is None:
            # Reshaped for PyTorch convolution: (out_channels, in_channels, *kernel_size)
            out_shape = (self.out_features, self.in_features, *self.kernel_size)
        else:
            out_shape = shape
        
        # Calculate the number of weights
        n_weight = np.prod(out_shape)
        
        # Sample modulus from Chi distribution with 4 degrees of freedom
        modulus = self.chi_distribution(n_weight, scale)
        
        # Generate random unit quaternion components
        # Create a random 3D unit vector
        v1 = self.rng.normal(0, 1, n_weight)
        v2 = self.rng.normal(0, 1, n_weight)
        v3 = self.rng.normal(0, 1, n_weight)
        
        # Normalize to unit vector
        norm = np.sqrt(v1**2 + v2**2 + v3**2) + 1e-8  # Avoid division by zero
        v1, v2, v3 = v1 / norm, v2 / norm, v3 / norm
        
        # Random angle for rotation
        theta = self.rng.uniform(-np.pi, np.pi, n_weight)
        
        # Construct quaternion components using quaternion rotation formula
        # w = cos(theta/2)
        # x,y,z = sin(theta/2) * unit_vector
        half_theta = theta / 2.0
        weight_r = np.cos(half_theta)
        weight_i = np.sin(half_theta) * v1
        weight_j = np.sin(half_theta) * v2
        weight_k = np.sin(half_theta) * v3
        
        # Apply modulus to get the final weights
        weight_r = modulus * weight_r
        weight_i = modulus * weight_i
        weight_j = modulus * weight_j
        weight_k = modulus * weight_k
        
        # Reshape to the desired output shape
        weight_r = weight_r.reshape(out_shape)
        weight_i = weight_i.reshape(out_shape)
        weight_j = weight_j.reshape(out_shape)
        weight_k = weight_k.reshape(out_shape)
        
        # Convert to PyTorch tensors and move to device if specified
        weight_r = torch.FloatTensor(weight_r)
        weight_i = torch.FloatTensor(weight_i)
        weight_j = torch.FloatTensor(weight_j)
        weight_k = torch.FloatTensor(weight_k)
        
        if device is not None:
            weight_r = weight_r.to(device)
            weight_i = weight_i.to(device)
            weight_j = weight_j.to(device)
            weight_k = weight_k.to(device)
            
        return weight_r, weight_i, weight_j, weight_k
    
    def chi_distribution(self, size, scale):
        """
        Generate samples from a Chi distribution with 4 degrees of freedom.
        
        This is used for the modulus of quaternion weights.
        
        Args:
            size: Number of samples to generate
            scale: Scaling factor for the distribution
            
        Returns:
            Numpy array of Chi-distributed random values
        """
        # Chi distribution with 4 DOF can be generated from normal distributions
        x1 = self.rng.normal(0, scale, size)
        x2 = self.rng.normal(0, scale, size)
        x3 = self.rng.normal(0, scale, size)
        x4 = self.rng.normal(0, scale, size)
        
        # The modulus is the Euclidean norm of these 4 normal variables
        return np.sqrt(x1**2 + x2**2 + x3**2 + x4**2)       