# models/model_builder.py

import torch.nn as nn
import yaml
from quaternion.conv import QConv, QConv2d
from quaternion.init import QInit
from .blocks.block import C3k2, SPPF, C2PSA , PSABlock, Reshape
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
    Initialize weights for a layer.
    """
    if hasattr(layer, 'weight') and layer.weight is not None:
        # Initialize standard layers
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if hasattr(layer, 'bias') and layer.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(layer.bias, -bound, bound)
    else:
        # Skip weight initialization for QConv and QConv2d layers
        pass

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
        'Reshape': Reshape,
        'QConv2d': QConv2d,
        'C3k2': C3k2,
        'PSABlock': PSABlock,
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
    Build a module from the configuration.

    Args:
        cfg (list): List of layer configurations.
        module_dict (dict): Mapping from layer names to their implementations.

    Returns:
        nn.Sequential: The constructed model layers.
    """
    layers = []
    for idx, layer_cfg in enumerate(cfg):
        if isinstance(layer_cfg, list):
            # Unpack layer configuration
            from_idx, num_repeats, module_name, module_args = layer_cfg

            # Process module arguments
            if isinstance(module_args, dict):
                args = module_args
            elif isinstance(module_args, list):
                # Convert list of key=value strings to dict
                args = {}
                for arg in module_args:
                    if isinstance(arg, str) and '=' in arg:
                        key, value = arg.split('=')
                        key = key.strip()
                        value = value.strip()
                        # Convert value to appropriate type
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            try:
                                value = int(value)
                            except ValueError:
                                try:
                                    value = float(value)
                                except ValueError:
                                    pass  # Keep as string
                        args[key] = value
            else:
                args = {}

            module_class = module_dict.get(module_name)
            if module_class is None:
                raise ValueError(f"Module '{module_name}' not found in module_dict.")

            # Instantiate the module
            for _ in range(num_repeats):
                module_instance = module_class(**args)
                initialize_weights(module_instance)
                layers.append(module_instance)
                # print(f"Added layer {idx}: {module_name} with args {args}")
        else:
            # Handle other configurations if necessary
            pass
    return nn.Sequential(*layers)
