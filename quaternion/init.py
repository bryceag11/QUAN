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