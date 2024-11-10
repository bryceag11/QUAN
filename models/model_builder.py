# models/model_builder.py

import torch.nn as nn
import yaml
from quaternion.conv import QConv, QConv2d
from quaternion.init import QInit
from .blocks.block import C3k2, SPPF, C2PSA 
from .neck.neck import QuaternionConcat, QuaternionFPN, QuaternionPAN
from .heads.qdet_head import QDetectHead, QOBBHead
import torch 
from data.transforms.quaternion import RGBtoQuatTransform


# models/model_builder.py

import torch.nn as nn
import math
import numpy as np
from quaternion import QInit


def initialize_weights(layer):
    """
    Initialize weights for a layer, including quaternion layers.
    """
    if isinstance(layer, QConv) or isinstance(layer, QConv2d):
        initializer = QInit(
            kernel_size=layer.kernel_size,
            input_dim=layer.in_channels,
            weight_dim=layer.rank
        )
        weight_dict = initializer.initialize(layer.modulus.shape, device=layer.modulus.device)
        layer.modulus.data = weight_dict['modulus']
        layer.phase.data = weight_dict['phase']
    elif hasattr(layer, 'weight') and layer.weight is not None:
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if hasattr(layer, 'bias') and layer.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(layer.bias, -bound, bound)

def load_model_from_yaml(config_path):
    """
    Load model architecture from YAML configuration.
    
    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        nn.Module: The constructed model.
        int: Number of classes (nc).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    nc = config['nc']
    backbone_cfg = config['backbone']
    head_cfg = config['head']
    
    # Define your module dictionary
    module_dict = {
        'QConv': QConv,
        'QConv2d': QConv2d,
        'C3k2': C3k2,
        'SPPF': SPPF,
        'C2PSA': C2PSA,
        'nn.Upsample': nn.Upsample,
        'QuaternionConcat': QuaternionConcat,
        'QOBBHead': QOBBHead,
        'QDetectHead': QDetectHead,
        # Add other layers as needed
    }
    
    # Build backbone
    backbone = build_from_cfg(backbone_cfg, module_dict)
    
    # Build head
    head = build_from_cfg(head_cfg, module_dict)
    
    # Combine backbone and head
    model = nn.Sequential(backbone, head)
    
    return model, nc

def build_from_cfg(cfg, module_dict):
    """
    Build a module from config.

    Args:
        cfg (list): List of layer configurations.
        module_dict (dict): Dictionary mapping layer names to their implementations.

    Returns:
        nn.Sequential: The constructed model layers.
    """
    layers = []
    for layer_cfg in cfg:
        if isinstance(layer_cfg, list):
            # Example: [-1, 1, QConv2d, [in_channels=3, out_channels=64, kernel_size=3, stride=2]]
            in_idx, num_repeat, layer_type, layer_args = layer_cfg
            if isinstance(layer_args, list):
                # Parse key=value strings into a dictionary
                args_dict = {}
                for arg in layer_args:
                    if isinstance(arg, str) and '=' in arg:
                        key, value = arg.split('=')
                        # Convert to appropriate type
                        try:
                            value = int(value)
                        except ValueError:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                        args_dict[key] = value
                    else:
                        # Handle positional arguments if any (not recommended)
                        pass
            else:
                args_dict = layer_args  # Assuming it's already a dict

            # Instantiate the layer with keyword arguments
            layer = module_dict[layer_type](**args_dict)

            # Apply weight initialization
            initialize_weights(layer)

            # Handle multiple repeats by creating new instances each time to avoid weight sharing
            for _ in range(num_repeat):
                # For each repeat, instantiate a new layer with the same arguments
                layer = module_dict[layer_type](**args_dict)
                initialize_weights(layer)
                layers.append(layer)
        elif isinstance(layer_cfg, dict):
            # Handle more complex configurations if needed
            pass
    return nn.Sequential(*layers)
