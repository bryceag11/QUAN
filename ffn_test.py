import torch
import torch.nn as nn
from quaternion.conv import QConv2d  # Ensure this path is correct
from models.blocks.block import MaxSigmoidAttnBlock, C2PSA, PSABlock  # Ensure these are correctly imported
import unittest

class TestQuaternionAttentionBlocks(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)  # For reproducibility

        # Common test dimensions
        self.batch_size = 2
        self.quaternion_dim = 4
        self.c_per_quaternion = 16  # C_per_quaternion = C_total / Q
        self.in_channels = self.c_per_quaternion * self.quaternion_dim  # 64
        self.spatial_size = 32

    def test_psablock(self):
        """Test PSABlock thoroughly."""
        print("\nTesting PSABlock...")

        # Create input tensor with quaternion dimension
        # Correct shape: [B, Q, C_per_quaternion, H, W]
        x = torch.randn(
            self.batch_size, 
            self.quaternion_dim,
            self.c_per_quaternion,
            self.spatial_size, 
            self.spatial_size
        ).to(self.device)
        print(f"Input shape: {x.shape}")

        block = PSABlock(
            c=self.in_channels,
            attn_ratio=1.0,
            num_heads=4,
            shortcut=True
        ).to(self.device)

        try:
            print("\nTesting full forward pass...")
            with torch.no_grad():
                output = block(x)
            print(f"Output shape: {output.shape}")

            expected_shape = x.shape
            self.assertEqual(output.shape, expected_shape, 
                             f"Expected output shape {expected_shape}, but got {output.shape}")

            # Print statistics
            print("\nOutput statistics:")
            print(f"  mean: {output.mean().item():.4f}")
            print(f"  std: {output.std().item():.4f}")
            print(f"  min: {output.min().item():.4f}")
            print(f"  max: {output.max().item():.4f}")

            # Check for NaN and Inf values
            self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
            self.assertFalse(torch.isinf(output).any(), "Output contains infinite values")

            print("✓ PSABlock test passed")

        except Exception as e:
            print(f"✗ PSABlock test failed: {str(e)}")
            raise e

    # Other test cases can be updated similarly...

def run_tests():
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
