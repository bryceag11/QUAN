# models/model_builder.py

import torch.nn as nn
import yaml
from quaternion.conv import QConv, QConv2D
from quaternion.init import QInit
from .blocks.block import C3k2, SPPF, C2PSA , PSABlock, Reshape, QAdaptiveFeatureExtraction, QDualAttention, QEnhancedDetectHead, QAdaptiveFusion
from .neck.neck import QuaternionConcat, QuaternionFPN, QuaternionPAN, QuaternionUpsample
from .heads.qdet_head import QDetectHead, QOBBHead, QuaternionPooling
import torch 
from data.transforms.quaternion import RGBtoQuatTransform
from quaternion.qbatch_norm import IQBN, QBN

# models/model_builder.py

import torch.nn as nn
import math
import numpy as np
from quaternion import QInit

# def init_weights(m):
#     if isinstance(m, QConv2D):
#         # Initialize the internal conv layers of QConv2D
#         nn.init.kaiming_normal_(m.conv_rr.weight, mode='fan_out', nonlinearity='relu')
#         if m.conv_rr.bias is not None:
#             nn.init.constant_(m.conv_rr.bias, 0)
            
#         # Initialize imaginary parts
#         for conv in [m.conv_ri, m.conv_rj, m.conv_rk]:
#             nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
#             if conv.bias is not None:
#                 nn.init.constant_(conv.bias, 0)
                
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
            
#     elif isinstance(m, IQBN):
#         # IQBN has gamma and beta parameters
#         if hasattr(m, 'gamma'):
#             nn.init.constant_(m.gamma, 1.0)
#         if hasattr(m, 'beta'):
#             nn.init.constant_(m.beta, 0.0)
    
#     elif isinstance(m, (QBN, nn.BatchNorm2d)):
#         if hasattr(m, 'weight') and m.weight is not None:
#             nn.init.constant_(m.weight, 1.0)
#         if hasattr(m, 'bias') and m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)

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
    neck_cfg = config.get('neck', [])  # Get neck config, empty list if not present
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
        'QuaternionPooling': QuaternionPooling,
        'QAdaptiveFeatureExtraction': QAdaptiveFeatureExtraction,
        'QDualAttention': QDualAttention,
        'QAdaptiveFusion': QAdaptiveFusion,
        'QEnhancedDetectHead': QEnhancedDetectHead,
        'QuaternionUpsample': QuaternionUpsample
        # Add other layers as needed
    }
    
    # Build backbone
    backbone = build_from_cfg(backbone_cfg, module_dict)[0]
    
    # Build neck with layer references
    neck_tuple = build_from_cfg(neck_cfg, module_dict)
    
    # Build head
    head = build_from_cfg(head_cfg, module_dict)[0]

    # Combine backbone and head
    model = CustomModel(backbone, neck_tuple, head)
    
    return model, nc

def build_from_cfg(cfg, module_dict):
    """Build a module from the configuration."""
    layers = []
    layer_refs = {}  # Store layer references
    
    for idx, layer_cfg in enumerate(cfg):
        if isinstance(layer_cfg, list):
            from_layer = layer_cfg[0]
            num_repeats = layer_cfg[1]
            module_name = layer_cfg[2]
            module_args = layer_cfg[3] if len(layer_cfg) > 3 else {}

            # Get module class
            module_class = module_dict.get(module_name)
            if module_class is None:
                raise ValueError(f"Module '{module_name}' not found in module_dict.")

            # Create module instance
            for _ in range(num_repeats):
                module_instance = module_class(**module_args)
                
                # Handle layer references for both neck and head
                if isinstance(from_layer, list):
                    if module_name in ['QuaternionConcat', 'QuaternionUpsample', 'QDetectHead']:
                        flat_refs = []
                        for ref in from_layer:
                            if isinstance(ref, list):
                                flat_refs.extend(ref)
                            else:
                                flat_refs.append(ref)
                        module_instance.from_layers = flat_refs
                        layer_refs[len(layers)] = flat_refs
                else:
                    module_instance.from_layers = [from_layer]
                    layer_refs[len(layers)] = [from_layer]
                
                layers.append(module_instance)

    print("Layer References (build_from_cfg):", layer_refs)
    return nn.ModuleList(layers), layer_refs


class CustomModel(nn.Module):
    def __init__(self, backbone: nn.ModuleList, neck_tuple: tuple, head: nn.ModuleList):
        super().__init__()
        self.backbone = backbone
        self.neck, self.neck_layer_refs = neck_tuple
        self.head = head
        self.feature_maps = {}
        
        # Debug: print layer references during initialization
        print("Neck Layer References:", self.neck_layer_refs)

    def forward(self, x):
        self.feature_maps = {}
        
        # Process backbone
        out = x
        for idx, layer in enumerate(self.backbone):
            out = layer(out)
            self.feature_maps[idx] = out

        # Process neck
        for idx, layer in enumerate(self.neck):
            refs = list(self.neck_layer_refs.get(idx, [-1]))
            
            if isinstance(layer, QuaternionConcat):
                input_features = [self.feature_maps[ref] if ref != -1 else out for ref in refs]
                out = layer(input_features)
            elif isinstance(layer, QuaternionUpsample):
                ref = refs[0]
                input_feature = self.feature_maps[ref] if ref != -1 else out
                out = layer(input_feature)
            else:
                out = layer(out)
            
            self.feature_maps[len(self.backbone) + idx] = out

        # Process head
        for idx, layer in enumerate(self.head):
            if isinstance(layer, QDetectHead):
                # Get the layer references from the YAML config
                layer_refs = getattr(layer, 'from_layers', None)
                if layer_refs:
                    # Collect features based on specified indices
                    head_features = []
                    for ref in layer_refs:
                        if ref in self.feature_maps:
                            head_features.append(self.feature_maps[ref])
                        else:
                            raise ValueError(f"Feature map {ref} not found")
                    
                    # Verify channel dimensions match expected
                    for feat, expected_ch in zip(head_features, layer.ch):
                        actual_ch = feat.size(1) * 4  # Account for quaternion channels
                        if actual_ch != expected_ch:
                            raise ValueError(
                                f"Channel mismatch: expected {expected_ch}, got {actual_ch}"
                            )
                    
                    return layer(head_features)
            out = layer(out)
        
        return out
    
