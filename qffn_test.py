import torch
import torch.nn as nn
import unittest

# Assuming QConv2d, IQBN, and QReLU are defined in the 'quaternion' module
# Replace 'quaternion.conv' and 'quaternion.nn' with the actual module paths if different
from quaternion.conv import QConv2d
from quaternion.qbatch_norm import IQBN, QBN, VQBN
from quaternion.qactivation import QReLU
from models.blocks.block import PSABlock, C2PSA, QAttention


class TestC2PSA(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment, including device configuration and test parameters.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)  # For reproducibility

        # Define test parameters
        self.batch_size = 2
        self.quaternion_dim = 4
        self.c_per_quaternion = 16  # C_per_quaternion = C_total / Q
        self.in_channels = self.c_per_quaternion * self.quaternion_dim  # Example: 16 * 4 = 64
        self.out_channels = 64  # Must be a multiple of 4
        self.n = 2  # Number of PSABlock modules
        self.e = 0.5  # Expansion ratio
        self.spatial_size = 32

    def test_c2psa_forward_pass(self):
        """
        Test the forward pass of C2PSA, capturing and printing outputs at each internal layer.
        """
        print("\n=== Testing C2PSA Forward Pass ===")

        # Initialize C2PSA
        c2psa = C2PSA(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n=self.n,
            e=self.e
        ).to(self.device)
        c2psa.eval()  # Set to evaluation mode to deactivate batch norm updates

        # Create a sample input tensor
        # Shape: [B, C, Q, H, W] = [2, 64, 4, 32, 32]
        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.quaternion_dim,
            self.spatial_size,
            self.spatial_size
        ).to(self.device)
        print(f"Sample Input Shape: {x.shape}")

        # Define hooks to capture outputs
        hook_outputs = {}

        def get_hook(name):
            def hook(module, input, output):
                print(f"\nOutput of {name}: {output.shape}")
                hook_outputs[name] = output
            return hook

        # Register hook for the initial convolution
        c2psa.cv1.register_forward_hook(get_hook('Initial Conv1 (cv1)'))

        # Register hook for the first batch normalization
        c2psa.bn1.register_forward_hook(get_hook('Initial BatchNorm1 (bn1)'))

        # Register hook for the first activation
        c2psa.act1.register_forward_hook(get_hook('Initial Activation1 (act1)'))

        # Register hooks for each PSABlock in the sequence
        for idx, layer in enumerate(c2psa.m):
            layer_name = f"PSABlock {idx + 1}"
            # Register a hook on the entire PSABlock to capture its output
            layer.register_forward_hook(get_hook(f"{layer_name} - Output"))
            # Optionally, register hooks inside PSABlock if needed
            # For simplicity, we capture the output of each PSABlock

        # Register hook for the final convolution
        c2psa.cv2.register_forward_hook(get_hook('Final Conv2 (cv2)'))

        # Register hook for the final batch normalization
        c2psa.bn2.register_forward_hook(get_hook('Final BatchNorm2 (bn2)'))

        # Register hook for the final activation
        c2psa.act2.register_forward_hook(get_hook('Final Activation2 (act2)'))

        try:
            # Perform the forward pass
            with torch.no_grad():
                output = c2psa(x)
            print(f"\nFinal Output Shape: {output.shape}")

            # Define the expected output shape
            expected_shape = x.shape  # [2, 64, 4, 32, 32]
            self.assertEqual(
                output.shape, expected_shape,
                f"Output shape {output.shape} does not match expected shape {expected_shape}"
            )

            # Check for numerical stability
            self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
            self.assertFalse(torch.isinf(output).any(), "Output contains infinite values")

            # Optionally, print output statistics
            print("\nOutput Statistics:")
            print(f"  Mean: {output.mean().item():.4f}")
            print(f"  Std Dev: {output.std().item():.4f}")
            print(f"  Min: {output.min().item():.4f}")
            print(f"  Max: {output.max().item():.4f}")

            print("\n✓ C2PSA Forward Pass Test Passed")

        except Exception as e:
            print(f"✗ C2PSA Forward Pass Test Failed: {str(e)}")
            raise e

    # You can add more test methods here if needed

def run_tests():
    """
    Run all the unit tests.
    """
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()