# backbone_test.py

import torch
import torch.nn as nn
from quaternion.conv import QConv2d
from models.blocks.block import QAttention, C2PSA, PSABlock, MaxSigmoidAttnBlock

def test_qattention_fixed():
    """Test the fixed Quaternion Attention module."""
    print("\n=== Testing QAttention (Fixed) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    channels = 256  # Must be divisible by num_heads
    spatial_size = 20
    num_heads = 8

    try:
        # Create input tensor with quaternion dimension
        x = torch.randn(batch_size, channels, 4, spatial_size, spatial_size).to(device)
        print(f"Input shape: {x.shape}")

        # Create QAttention module
        attn = QAttention(
            dim=channels,
            num_heads=num_heads,
            attn_ratio=1.0
        ).to(device)

        # Forward pass
        with torch.no_grad():
            output = attn(x)
        
        # Expected shape should match input shape
        expected_shape = (batch_size, channels, 4, spatial_size, spatial_size)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert output.shape == expected_shape, \
            f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

        # Print statistics
        print("\nOutput statistics:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        
        # Test with different batch sizes and spatial dimensions
        test_shapes = [
            (1, channels, 4, 16, 16),
            (4, channels, 4, 32, 32),
            (2, channels, 4, 24, 24),
        ]
        
        print("\nTesting different input shapes:")
        for shape in test_shapes:
            x = torch.randn(*shape).to(device)
            with torch.no_grad():
                output = attn(x)
            print(f"Input shape: {shape} -> Output shape: {output.shape}")
            assert output.shape == shape, f"Shape mismatch for {shape}"
        
        print("✓ QAttention test passed")
        return True

    except Exception as e:
        print("✗ QAttention test failed")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_c2psa_fixed():
    """Test C2PSA module with proper quaternion dimensions."""
    print("\n=== Testing C2PSA ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    in_channels = 1024
    spatial_size = 20

    try:
        # Create input tensor with quaternion dimension
        x = torch.randn(batch_size, in_channels, 4, spatial_size, spatial_size).to(device)
        print(f"Input shape: {x.shape}")

        # Create C2PSA module
        c2psa = C2PSA(
            in_channels=in_channels,
            out_channels=in_channels,
            n=1,  # Number of PSA blocks
            e=0.5,  # Expansion ratio
            g=1,  # Groups
            shortcut=True
        ).to(device)

        # Forward pass
        with torch.no_grad():
            output = c2psa(x)
        
        # Expected shape should match input shape
        expected_shape = (batch_size, in_channels, 4, spatial_size, spatial_size)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected_shape}")
        
        assert output.shape == expected_shape, \
            f"Shape mismatch: got {output.shape}, expected {expected_shape}"

        # Check for NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"

        # Print statistics
        print("\nOutput statistics:")
        print(f"  mean: {output.mean().item():.4f}")
        print(f"  std: {output.std().item():.4f}")
        print(f"  min: {output.min().item():.4f}")
        print(f"  max: {output.max().item():.4f}")
        
        # Test with different batch sizes and spatial dimensions
        test_shapes = [
            (1, in_channels, 4, 16, 16),
            (4, in_channels, 4, 32, 32),
            (2, in_channels, 4, 24, 24),
        ]
        
        print("\nTesting different input shapes:")
        for shape in test_shapes:
            x = torch.randn(*shape).to(device)
            with torch.no_grad():
                output = c2psa(x)
            print(f"Input shape: {shape} -> Output shape: {output.shape}")
            assert output.shape == shape, f"Shape mismatch for {shape}"
        
        print("✓ C2PSA test passed")
        return True

    except Exception as e:
        print("✗ C2PSA test failed")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all quaternion attention and C2PSA tests."""
    try:
        print("Starting Quaternion Attention and C2PSA tests...")
        
        qattn_success = test_qattention_fixed()
        if not qattn_success:
            print("QAttention tests failed, skipping C2PSA test")
            return
            
        c2psa_success = test_c2psa_fixed()
        
        if qattn_success and c2psa_success:
            print("\nAll tests completed successfully! ✓")
        else:
            print("\nSome tests failed! ✗")
            
    except Exception as e:
        print(f"\nTests failed with unexpected error! ✗")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_all_tests()
