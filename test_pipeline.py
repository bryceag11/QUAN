# test_pipeline.py

import torch
import yaml
import math
from data.dataloader import get_quaternion_dataloader
from data.transforms.quaternion import RGBtoQuatTransform
from models.model_builder import load_model_from_yaml
import torch.nn as nn
from models.neck.neck import QuaternionConcat, QuaternionFPN, QuaternionPAN
from quaternion.conv import QConv2D
from models.blocks.block import C3k2, SPPF, QBottleneck, QAdaptiveFeatureExtraction, QuaternionUpsample
from pathlib import Path

def print_tensor_stats(tensor, name="Tensor"):
    """Helper function to print tensor statistics."""
    print(f"\n{name} Statistics:")
    print(f"Shape: {tensor.shape}")
    print(f"dtype: {tensor.dtype}")
    print(f"device: {tensor.device}")
    if tensor.numel() > 0:
        print(f"Mean: {tensor.mean().item():.4f}")
        print(f"Std: {tensor.std().item():.4f}")
        print(f"Min: {tensor.min().item():.4f}")
        print(f"Max: {tensor.max().item():.4f}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

def test_data_pipeline(config_path='configs/default_config.yaml', model_config='configs/models/q.yaml'):
    """Enhanced test pipeline with better debugging and tracking."""
    
    print("\n" + "="*50)
    print("Starting Enhanced Test Pipeline")
    print("="*50)

    # Load configs
    print("\n=== Loading Configurations ===")
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    with open(model_config, 'r') as f:
        model_yaml = yaml.safe_load(f)
    
    # Build layer mapping
    layer_mapping = {}
    current_idx = 0
    
    def build_layer_mapping(config, section_name):
        nonlocal current_idx
        for layer in config:
            if isinstance(layer, list):
                from_layer = layer[0]
                repeats = layer[1]
                module_type = layer[2]
                args = layer[3] if len(layer) > 3 else {}
                
                layer_mapping[current_idx] = {
                    'from': from_layer,
                    'type': module_type,
                    'args': args,
                    'section': section_name,
                    'idx': current_idx
                }
                current_idx += 1
    
    # Build complete layer mapping
    build_layer_mapping(model_yaml['backbone'], 'backbone')
    build_layer_mapping(model_yaml['head'], 'head')
    
    print("\n=== Layer Mapping ===")
    for idx, info in layer_mapping.items():
        print(f"Layer {idx} ({info['section']}): {info['type']} from {info['from']}")
    
    # Get dataset info
    active_dataset = data_config['active_dataset']
    train_info = data_config['datasets'][active_dataset]['train']
    
    # Create transform and dataloader
    transform = RGBtoQuatTransform(real_component=1.0)
    dataloader = get_quaternion_dataloader(
        img_dir=train_info['img_dir'],
        ann_file=train_info['ann_file'],
        batch_size=2,
        shuffle=False,
        num_workers=0,
        dataset_type=active_dataset.lower(),
        transform=transform
    )
    
    # Load model
    print("\n=== Loading Model ===")
    model, nc = load_model_from_yaml(model_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Track activation shapes
    activation_shapes = {}
    
    def get_layer_info(module):
        """Find layer info in mapping."""
        found_layers = [idx for idx, info in layer_mapping.items() 
                       if info['type'] == module.__class__.__name__]
        if found_layers:
            return [layer_mapping[idx] for idx in found_layers]
        return None
    
    def hook_fn(module, input, output):
        """Enhanced forward hook with detailed logging."""
        module_name = module.__class__.__name__
        layer_info = get_layer_info(module)
        
        print("\n" + "="*50)
        if layer_info:
            for info in layer_info:
                print(f"Layer: {module_name}")
                print(f"YAML Position: {info['idx']} ({info['section']})")
                print(f"Takes input from: {info['from']}")
                print(f"Arguments: {info['args']}")
        else:
            print(f"Layer: {module_name} (Auxiliary layer)")
        
        # Input analysis
        print("\nInput Analysis:")
        if input is None or len(input) == 0:
            print("No input provided")
        else:
            if isinstance(input[0], (list, tuple)):
                for i, inp in enumerate(input[0]):
                    if inp is not None:
                        print(f"Input {i}:")
                        print_tensor_stats(inp, f"Input tensor {i}")
                    else:
                        print(f"Input {i} is None")
            else:
                if input[0] is not None:
                    print_tensor_stats(input[0], "Input tensor")
                else:
                    print("Input is None")
        
        # Output analysis
        print("\nOutput Analysis:")
        if output is None:
            print("Warning: Module output is None")
            return
            
        if isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if out is not None:
                    print(f"Output {i}:")
                    print_tensor_stats(out, f"Output tensor {i}")
                else:
                    print(f"Output {i} is None")
        else:
            print_tensor_stats(output, "Output tensor")
        
        # Store activation shape
        if layer_info:
            for info in layer_info:
                activation_shapes[info['idx']] = {
                    'input': input[0].shape if not isinstance(input[0], (list, tuple)) else [t.shape for t in input[0]],
                    'output': output.shape,
                    'type': module_name
                }
        
        print("="*50)
    
    # Register hooks for all relevant layer types
    hook_types = (
        QuaternionConcat, QConv2D, nn.BatchNorm2d, nn.ReLU, nn.Upsample,
        C3k2, SPPF, QuaternionFPN, QuaternionPAN, QBottleneck,
        QAdaptiveFeatureExtraction, QuaternionUpsample
    )
    
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, hook_types):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    print("\n=== Starting Forward Pass Tests ===")
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 1:  # Test only first two batches
                break
            
            print(f"\nProcessing Batch {batch_idx}")
            print("="*30)
            
            # Move data to device
            images = batch['image'].to(device)
            print_tensor_stats(images, "Input batch")
            
            # Forward pass
            try:
                with torch.no_grad():
                    outputs = model(images)
                
                print("\nForward pass successful!")
                for i, group in enumerate(outputs):
                    print(f"\nGroup {i}:")
                    if isinstance(group, (list, tuple)):
                        print("\n Pred Output:")
                        for idx, output in enumerate(group):
                            print_tensor_stats(output, f"Model output {idx}")
                    else:
                        print("\n Anchor Output:")
                        for idx, output in group.items():
                            print_tensor_stats(output, f"Model output {idx}")
                    

                # print("\nForward pass successful!")
                # for i, group in enumerate(outputs):
                #     print(f"\nGroup {i}:")
                #     if isinstance(group, dict):  # Handle dictionary outputs
                #         for key, tensor in group.items():
                #             print(f"Processing Group {i}, Key {key}")
                #             if hasattr(tensor, 'shape'):  # Check if it's a tensor
                #                 print(f"Shape of tensor at Group {i}, Key {key}: {tensor.shape}")
                #             else:
                #                 print(f"Value at Group {i}, Key {key} is not a tensor.")
                #     elif isinstance(group, (list, tuple)):  # Handle nested lists or tuples
                #         for j, output in enumerate(group):
                #             print(f"Processing Group {i}, Output {j}")
                #             if hasattr(output, 'shape'):
                #                 print(f"Shape of output {i}-{j}: {output.shape}")
                #             else:
                #                 print(f"Output {i}-{j} is not a tensor.")
                #     else:  # Handle other cases (e.g., direct tensors)
                #         if hasattr(group, 'shape'):
                #             print(f"Shape of Group {i}: {group.shape}")
                #         else:
                #             print(f"Group {i} is not a tensor.")
            except Exception as e:
                print("\nError during forward pass!")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                
                # Print last successful layer
                print("\nLast successful layer activations:")
                last_layer = max(activation_shapes.keys())
                print(f"Layer {last_layer} ({activation_shapes[last_layer]['type']}):")
                print(f"Input shape: {activation_shapes[last_layer]['input']}")
                print(f"Output shape: {activation_shapes[last_layer]['output']}")
                
                raise e
            
            # Memory usage
            if torch.cuda.is_available():
                print(f"\nGPU Memory:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB")
                print(f"Cached: {torch.cuda.memory_reserved()/1e6:.1f}MB")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Save activation shapes to file
        save_dir = Path('debug_outputs')
        save_dir.mkdir(exist_ok=True)
        
        with open(save_dir / 'activation_shapes.txt', 'w') as f:
            f.write("Layer Activation Shapes:\n")
            for idx in sorted(activation_shapes.keys()):
                info = activation_shapes[idx]
                f.write(f"\nLayer {idx} ({info['type']}):\n")
                f.write(f"Input shape: {info['input']}\n")
                f.write(f"Output shape: {info['output']}\n")
                f.write("-"*30 + "\n")

if __name__ == "__main__":
    test_data_pipeline()