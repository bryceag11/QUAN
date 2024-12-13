# models/model_builder.py

import torch.nn as nn
import yaml
from quaternion.conv import QConv, QConv2D
from quaternion.init import QInit
from .blocks.block import C3k2, SPPF, C2PSA , PSABlock, Reshape, QAdaptiveFeatureExtraction, QDualAttention, QEnhancedDetectHead, QAdaptiveFusion, QuaternionUpsample
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
        # Skip weight initialization for QConv and QConv2D layers
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
        'QConv2D': QConv2D,
        'C3k2': C3k2,
        'PSABlock': PSABlock,
        'SPPF': SPPF,
        'C2PSA': C2PSA,
        'nn.Upsample': nn.Upsample,
        'QuaternionConcat': QuaternionConcat,
        'QOBBHead': QOBBHead,
        'QDetectHead': QDetectHead,
        'QAdaptiveFeatureExtraction': QAdaptiveFeatureExtraction,
        'QDualAttention': QDualAttention,
        'QAdaptiveFusion': QAdaptiveFusion,
        'QEnhancedDetectHead': QEnhancedDetectHead,
        'QuaternionUpsample': QuaternionUpsample
        # Add other layers as needed
    }
    
    # Build backbone
    backbone = build_from_cfg(backbone_cfg, module_dict)
    
    # Build head
    head = build_from_cfg(head_cfg, module_dict)
    print("\n=== Head Layers ===")
    for idx, layer in enumerate(head):
        print(f"Layer {idx}: {layer.__class__.__name__}")
    # Combine backbone and head
    model = CustomModel(backbone, head)
    
    return model, nc

def build_from_cfg(cfg, module_dict):
    """
    Build a module from the configuration.

    Args:
        cfg (list): List of layer configurations.
        module_dict (dict): Mapping from layer names to their implementations.

    Returns:
        nn.ModuleList: The constructed model layers.
    """
    layers = []
    for idx, layer_cfg in enumerate(cfg):
        if isinstance(layer_cfg, list):
            # Handle different layer config formats
            if isinstance(layer_cfg[0], list):
                # Format: [[layer_indices], repeats, module_name, args]
                from_layers = layer_cfg[0]
                num_repeats = layer_cfg[1]
                module_name = layer_cfg[2]
                module_args = layer_cfg[3] if len(layer_cfg) > 3 else {}
            else:
                # Format: [from_layer, repeats, module_name, args]
                from_layers = [layer_cfg[0]]
                num_repeats = layer_cfg[1]
                module_name = layer_cfg[2]
                module_args = layer_cfg[3] if len(layer_cfg) > 3 else {}

            # Get module class
            module_class = module_dict.get(module_name)
            if module_class is None:
                raise ValueError(f"Module '{module_name}' not found in module_dict.")

            # Process arguments
            if isinstance(module_args, dict):
                args = module_args
            else:
                args = {}

            # Create module instance
            for _ in range(num_repeats):
                module_instance = module_class(**args)
                # Store from_layers information for concatenation layers
                if isinstance(module_instance, QuaternionConcat):
                    module_instance.from_layers = from_layers
                initialize_weights(module_instance)
                layers.append(module_instance)

    return nn.ModuleList(layers)


class CustomModel(nn.Module):
    def __init__(self, backbone: nn.ModuleList, head: nn.ModuleList):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        feature_maps = []
        detect_input_features = []
        first_conv = second_conv = final_c3k2 = None
        
        # Process backbone
        out = x
        for layer in self.backbone:
            out = layer(out)
            feature_maps.append(out)
            
        # Process head
        for layer in self.head:
            if isinstance(layer, (QDetectHead, QOBBHead)):
                if first_conv is None or second_conv is None or final_c3k2 is None:
                    raise ValueError("Missing required features for detection head")
                detect_inputs = [first_conv, second_conv, final_c3k2]

                return layer(detect_inputs)  # Returns predictions and anchor points
            
            # Process current layer
            if isinstance(layer, QuaternionConcat):
                if hasattr(layer, 'from_layers'):
                    input_features = []
                    for ref in layer.from_layers:
                        if isinstance(ref, list):
                            for r in ref:
                                input_features.append(feature_maps[r] if r != -1 else out)
                        else:
                            input_features.append(feature_maps[ref] if ref != -1 else out)
                    out = layer(input_features)
                else:
                    out = layer([out])
            else:
                out = layer(out)
                if isinstance(layer, QConv2D):
                    if first_conv is None:
                        first_conv = out
                    elif second_conv is None:
                        second_conv = out
                elif isinstance(layer, C3k2) and second_conv is not None:
                    final_c3k2 = out
            
            feature_maps.append(out)
        
        return out