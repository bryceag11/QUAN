import torch
import torch.nn as nn
from quaternion.conv import QConv2d
from quaternion.init import QInit
from models.blocks.block import C3k2, SPPF, C2PSA
from models.neck.neck import QuaternionConcat, QuaternionFPN, QuaternionPAN
from models.heads.qdet_head import QDetectHead, QOBBHead
import math

def test_layer(layer, input_tensor, layer_name=""):
    """Test a single layer with detailed debugging information."""
    print(f"\n=== Testing {layer_name} ===")
    print(f"Input shape: {input_tensor.shape}")
    
    try:
        # Print layer information
        if isinstance(layer, QConv2d):
            print(f"Layer info:")
            print(f"  in_channels: {layer.in_channels}")
            print(f"  out_channels: {layer.out_channels}")
            print(f"  kernel_size: {layer.kernel_size}")
            print(f"  Weight shapes:")
            print(f"    modulus: {layer.modulus.shape}")
            print(f"    phase: {layer.phase.shape}")
        
        # Forward pass
        output = layer(input_tensor)
        print(f"Output shape: {output.shape}")
        print(f"Output stats:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        print("Test PASSED ✓")
        return output
    
    except Exception as e:
        print(f"Test FAILED ✗")
        print(f"Error: {str(e)}")
        raise e

def test_backbone_components():
    """Test each component that would be in the backbone."""
    print("\n=== Testing Backbone Components ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create input tensor
    batch_size = 2
    input_size = 640
    x = torch.randn(batch_size, 4, input_size, input_size).to(device)
    print(f"Initial input shape: {x.shape}")

    # Test the model components one by one
    test_cases = [
        {
            'name': 'Initial Conv',
            'layer': QConv2d(4, 64, kernel_size=3, stride=2, padding=1),
            'expected_shape': (batch_size, 64, input_size//2, input_size//2)
        },
        {
            'name': 'Second Conv',
            'layer': QConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            'expected_shape': (batch_size, 128, input_size//4, input_size//4)
        }
    ]

    for test_case in test_cases:
        try:
            layer = test_case['layer'].to(device)
            name = test_case['name']
            expected_shape = test_case['expected_shape']
            
            print(f"\nTesting {name}")
            print("Input shape:", x.shape)
            print("Layer info:")
            print(f"  in_channels: {layer.in_channels}")
            print(f"  out_channels: {layer.out_channels}")
            print(f"  kernel_size: {layer.kernel_size}")
            print(f"  Weight shapes:")
            print(f"    modulus: {layer.modulus.shape}")
            print(f"    phase: {layer.phase.shape}")
            
            # Forward pass
            x = layer(x)
            
            print("Output shape:", x.shape)
            print("Expected shape:", expected_shape)
            assert x.shape == expected_shape, f"Shape mismatch: got {x.shape}, expected {expected_shape}"
            print("✓ Shape test passed")
            
            # Test output statistics
            print("Output statistics:")
            print(f"  mean: {x.mean().item():.4f}")
            print(f"  std: {x.std().item():.4f}")
            print(f"  min: {x.min().item():.4f}")
            print(f"  max: {x.max().item():.4f}")
            
        except Exception as e:
            print(f"✗ Test failed for {name}")
            print(f"Error: {str(e)}")
            raise e

def test_neck_components():
    """Test neck components including upsampling and concatenation."""
    print("\n=== Testing Neck Components ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # Create dummy feature maps of different sizes
    p3 = torch.randn(batch_size, 256, 80, 80).to(device)
    p4 = torch.randn(batch_size, 512, 40, 40).to(device)
    p5 = torch.randn(batch_size, 1024, 20, 20).to(device)

    test_cases = [
        {
            'name': 'P5 Upsampling',
            'layer': nn.Upsample(scale_factor=2, mode='nearest'),
            'input': p5,
            'expected_shape': (batch_size, 1024, 40, 40)
        },
        {
            'name': 'Feature Concatenation',
            'layer': torch.cat,
            'inputs': [p4, p5],
            'expected_shape': (batch_size, 1536, 40, 40)  # 512 + 1024 channels
        }
    ]

    for test_case in test_cases:
        try:
            name = test_case['name']
            print(f"\nTesting {name}")
            
            if name == 'Feature Concatenation':
                # Handle concatenation specially
                p5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')(test_case['inputs'][1])
                inputs = [test_case['inputs'][0], p5_upsampled]
                output = torch.cat(inputs, dim=1)
            else:
                output = test_case['layer'](test_case['input'])

            expected_shape = test_case['expected_shape']
            
            print(f"Output shape: {output.shape}")
            print(f"Expected shape: {expected_shape}")
            assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
            
            print("Output statistics:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")
            print("✓ Test passed")
            
        except Exception as e:
            print(f"✗ Test failed for {name}")
            print(f"Error: {str(e)}")
            raise e

def test_c3k2_block():
    """Test C3k2 block specifically."""
    print("\n=== Testing C3k2 Block ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    input_channels = 128
    output_channels = 256
    input_size = 160

    # Create input tensor
    x = torch.randn(batch_size, input_channels, input_size, input_size).to(device)
    print(f"Input shape: {x.shape}")

    try:
        # Create C3k2 block
        c3k2 = C3k2(
            in_channels=input_channels,
            out_channels=output_channels,
            n=2,  # Number of bottleneck blocks
            e=0.5,  # Expansion ratio
            g=1,  # Groups
            shortcut=True
        ).to(device)

        # Test each sub-component of C3k2
        # First branch: cv1
        y1 = c3k2.cv1(x)
        print(f"cv1 output shape: {y1.shape}")

        # Second branch: cv2
        y2 = c3k2.cv2(x)
        print(f"cv2 output shape: {y2.shape}")

        # Process through bottleneck blocks
        y3 = c3k2.m(y1)
        print(f"Bottleneck blocks output shape: {y3.shape}")

        # Concatenate and final convolution
        y = torch.cat([y2, y1, y3], dim=1)
        print(f"Concatenated shape: {y.shape}")
        
        output = c3k2.cv3(y)
        print(f"Final output shape: {output.shape}")
        
        expected_shape = (batch_size, output_channels, input_size, input_size)
        assert output.shape == expected_shape, f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        
        print("\nOutput statistics:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        print("✓ C3k2 test passed")

    except Exception as e:
        print("✗ C3k2 test failed")
        print(f"Error: {str(e)}")
        raise e


def test_initialization():
    """Test quaternion weight initialization."""
    print("\n=== Testing Weight Initialization ===")
    
    try:
        # Test different layer configurations
        test_configs = [
            (4, 64, 3),
            (64, 128, 3),
            (128, 256, 3),
        ]

        for in_ch, out_ch, kernel_size in test_configs:
            print(f"\nTesting initialization with:")
            print(f"in_channels={in_ch}, out_channels={out_ch}, kernel_size={kernel_size}")
            
            # Create layer
            layer = QConv2d(in_ch, out_ch, kernel_size)
            
            # Initialize weights using QInit
            initializer = QInit(
                kernel_size=layer.kernel_size,
                input_dim=layer.in_channels,
                weight_dim=2
            )
            
            # Get initialization
            weight_dict = initializer.initialize(
                shape=(layer.out_channels, layer.in_channels) + layer.kernel_size,
                device=layer.modulus.device
            )
            
            # Print statistics
            print("Modulus stats:")
            print(f"  mean: {weight_dict['modulus'].mean().item():.4f}")
            print(f"  std: {weight_dict['modulus'].std().item():.4f}")
            print("Phase stats:")
            print(f"  mean: {weight_dict['phase'].mean().item():.4f}")
            print(f"  std: {weight_dict['phase'].std().item():.4f}")

    except Exception as e:
        print("Initialization testing failed")
        raise e

def test_fpn():
    """Test Feature Pyramid Network (FPN) functionality."""
    print("\n=== Testing FPN ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # Create mock backbone feature maps with proper quaternion channel counts
    features = [
        torch.randn(batch_size, 256, 80, 80).to(device),  # P3
        torch.randn(batch_size, 512, 40, 40).to(device),  # P4
        torch.randn(batch_size, 1024, 20, 20).to(device)  # P5
    ]

    try:
        # Create FPN
        fpn = QuaternionFPN(
            in_channels=[256, 512, 1024],
            out_channels=256
        ).to(device)
        
        print("Input feature shapes:")
        for i, feat in enumerate(features):
            print(f"P{i+3}: {feat.shape}")

        # Forward pass through FPN
        fpn_outputs = fpn(features)
        
        print("\nFPN output shapes:")
        for i, out in enumerate(fpn_outputs):
            print(f"P{i+3}: {out.shape}")
            
        # Expected shapes (all with 256 channels)
        expected_shapes = [
            (batch_size, 256, 80, 80),  # P3
            (batch_size, 256, 40, 40),  # P4
            (batch_size, 256, 20, 20)   # P5
        ]
        
        # Verify shapes
        for output, expected_shape in zip(fpn_outputs, expected_shapes):
            assert output.shape == expected_shape, \
                f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        
        print("✓ FPN test passed")

    except Exception as e:
        print("✗ FPN test failed")
        print(f"Error: {str(e)}")
        raise e

def test_pan():
    """Test Path Aggregation Network (PAN) functionality."""
    print("\n=== Testing PAN ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # Create mock FPN outputs
    fpn_features = [
        torch.randn(batch_size, 256, 80, 80).to(device),  # P3
        torch.randn(batch_size, 256, 40, 40).to(device),  # P4
        torch.randn(batch_size, 256, 20, 20).to(device)   # P5
    ]

    try:
        # Create PAN
        pan = QuaternionPAN(
            in_channels=[256, 256, 256],
            out_channels=256
        ).to(device)
        
        print("Input feature shapes:")
        for i, feat in enumerate(fpn_features):
            print(f"P{i+3}: {feat.shape}")

        # Forward pass through PAN
        pan_outputs = pan(fpn_features)
        
        print("\nPAN output shapes:")
        for i, out in enumerate(pan_outputs):
            print(f"P{i+3}: {out.shape}")
            
        # Expected shapes
        expected_shapes = [
            (batch_size, 256, 80, 80),  # P3
            (batch_size, 256, 40, 40),  # P4
            (batch_size, 256, 20, 20)   # P5
        ]
        
        # Verify shapes and stats
        for i, (output, expected_shape) in enumerate(zip(pan_outputs, expected_shapes)):
            print(f"\nP{i+3} statistics:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")
            
            assert output.shape == expected_shape, \
                f"Shape mismatch: got {output.shape}, expected {expected_shape}"
        
        print("✓ PAN test passed")

    except Exception as e:
        print("✗ PAN test failed")
        print(f"Error: {str(e)}")
        raise e

def test_full_neck():
    """Test the full neck (FPN + PAN) functionality."""
    print("\n=== Testing Full Neck (FPN + PAN) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    try:
        # Create mock backbone features
        backbone_features = [
            torch.randn(batch_size, 256, 80, 80).to(device),  # P3
            torch.randn(batch_size, 512, 40, 40).to(device),  # P4
            torch.randn(batch_size, 1024, 20, 20).to(device)  # P5
        ]

        # Create FPN and PAN
        fpn = QuaternionFPN(
            in_channels=[256, 512, 1024],
            out_channels=256
        ).to(device)
        
        pan = QuaternionPAN(
            in_channels=[256, 256, 256],
            out_channels=256
        ).to(device)

        # Full forward pass
        fpn_outputs = fpn(backbone_features)
        final_outputs = pan(fpn_outputs)

        # Verify final output shapes
        expected_shapes = [
            (batch_size, 256, 80, 80),  # P3
            (batch_size, 256, 40, 40),  # P4
            (batch_size, 256, 20, 20)   # P5
        ]

        print("\nFinal output shapes and statistics:")
        for i, (output, expected_shape) in enumerate(zip(final_outputs, expected_shapes)):
            print(f"\nOutput {i} (P{i+3}):")
            print(f"Shape: {output.shape} (expected: {expected_shape})")
            print(f"Statistics:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")
            
            assert output.shape == expected_shape, \
                f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        print("✓ Full neck test passed")

    except Exception as e:
        print("✗ Full neck test failed")
        print(f"Error: {str(e)}")
        raise e

def test_detection_heads():
    """Test both detection and OBB heads."""
    print("\n=== Testing Detection Heads ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    # Create mock feature maps with quaternion dimension
    features = [
        torch.randn(batch_size, 256, 4, 80, 80).to(device),  # P3 features
        torch.randn(batch_size, 512, 4, 40, 40).to(device),  # P4 features
        torch.randn(batch_size, 1024, 4, 20, 20).to(device)  # P5 features
    ]

    try:
        # Test Detection Head
        print("\nTesting QDetectHead:")
        detect_head = QDetectHead(nc=80, ch=[256, 512, 1024]).to(device)
        detect_outputs = detect_head(features)
        
        for i, output in enumerate(detect_outputs):
            print(f"\nP{i+3} output:")
            print(f"Shape: {output.shape}")
            print(f"Stats:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")
        
        # Test OBB Head
        print("\nTesting QOBBHead:")
        obb_head = QOBBHead(nc=80, ch=[256, 512, 1024]).to(device)
        obb_outputs = obb_head(features)
        
        for i, output in enumerate(obb_outputs):
            print(f"\nP{i+3} output:")
            print(f"Shape: {output.shape}")
            print(f"Stats:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")
        
        print("\n✓ All head tests passed!")

    except Exception as e:
        print(f"\n✗ Head tests failed!")
        print(f"Error: {str(e)}")
        raise e

def test_sppf():
    """Test Spatial Pyramid Pooling - Fast (SPPF) module."""
    print("\n=== Testing SPPF ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    in_channels = 1024  # Common channel size where SPPF is used
    out_channels = 1024
    spatial_size = 20   # Common spatial size at this stage

    try:
        # Create input tensor
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
        print(f"Input shape: {x.shape}")

        # Create SPPF module
        sppf = SPPF(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5
        ).to(device)

        # Forward pass
        output = sppf(x)
        
        # Expected shape
        expected_shape = (batch_size, out_channels, spatial_size, spatial_size)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert output.shape == expected_shape, \
            f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        # Print statistics
        print("\nOutput statistics:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        
        print("✓ SPPF test passed")

    except Exception as e:
        print("✗ SPPF test failed")
        print(f"Error: {str(e)}")
        raise e

def test_c2psa():
    """Test C2PSA (Channel-wise Position Sensitive Attention) module."""
    print("\n=== Testing C2PSA ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    in_channels = 1024
    out_channels = 1024
    spatial_size = 20

    try:
        # Create input tensor
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
        print(f"Input shape: {x.shape}")

        # Create mock guide tensor if needed by your implementation
        guide = torch.randn(batch_size, 512).to(device)  # Adjust size based on your implementation
        
        # Create C2PSA module
        c2psa = C2PSA(
            in_channels=in_channels,
            out_channels=out_channels,
            n=1  # Number of PSA blocks
        ).to(device)

        # Test each component of C2PSA
        print("\nTesting C2PSA components:")
        
        # Initial split and convolution
        y = c2psa.cv1(x)
        print(f"After cv1 shape: {y.shape}")

        # First path (PSA blocks)
        if hasattr(c2psa, 'm'):
            y_psa = c2psa.m(y)
            print(f"After PSA blocks shape: {y_psa.shape}")

        # Forward pass with guide if needed
        if 'guide' in inspect.signature(c2psa.forward).parameters:
            output = c2psa(x, guide)
        else:
            output = c2psa(x)
        
        # Expected shape
        expected_shape = (batch_size, out_channels, spatial_size, spatial_size)
        
        print(f"\nFinal output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert output.shape == expected_shape, \
            f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        # Print statistics
        print("\nOutput statistics:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        
        print("✓ C2PSA test passed")

    except Exception as e:
        print("✗ C2PSA test failed")
        print(f"Error: {str(e)}")
        raise e

def test_integration():
    """Test SPPF and C2PSA integration."""
    print("\n=== Testing SPPF + C2PSA Integration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    in_channels = 1024
    spatial_size = 20

    try:
        # Create input tensor
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
        print(f"Input shape: {x.shape}")

        # Create modules
        sppf = SPPF(in_channels=in_channels, out_channels=in_channels).to(device)
        c2psa = C2PSA(in_channels=in_channels, out_channels=in_channels).to(device)

        # Forward pass through SPPF
        sppf_out = sppf(x)
        print(f"After SPPF shape: {sppf_out.shape}")

        # Forward pass through C2PSA
        final_out = c2psa(sppf_out)
        print(f"After C2PSA shape: {final_out.shape}")

        # Expected shape
        expected_shape = (batch_size, in_channels, spatial_size, spatial_size)
        assert final_out.shape == expected_shape, \
            f"Shape mismatch: got {final_out.shape}, expected {expected_shape}"

        # Print statistics
        print("\nFinal output statistics:")
        print(f"  mean: {final_out.mean().item():.4f}")
        print(f"  std: {final_out.std().item():.4f}")
        print(f"  min: {final_out.min().item():.4f}")
        print(f"  max: {final_out.max().item():.4f}")
        
        print("✓ Integration test passed")

    except Exception as e:
        print("✗ Integration test failed")
        print(f"Error: {str(e)}")
        raise e

if __name__ == "__main__":
    import inspect  # For inspecting function signatures
    
    try:
        print("Starting SPPF and C2PSA tests...")
        test_sppf()
        test_c2psa()
        test_integration()
        print("\nAll tests completed successfully! ✓")
    except Exception as e:
        print(f"\nTests failed! ✗")
        print(f"Error: {str(e)}")

