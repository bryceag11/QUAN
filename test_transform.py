import torch
import numpy as np
from PIL import Image
from data.transforms.quaternion import RGBtoQuatTransform

def test_transform():
    # Create a sample RGB image
    rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(rgb_image)
    
    # Create transform
    transform = RGBtoQuatTransform(real_component=1.0)
    
    print("=== Testing RGBtoQuatTransform ===")
    
    # Test with PIL Image
    print("\nTesting with PIL Image:")
    output = transform(pil_image)
    print(f"Input shape (HWC): {rgb_image.shape}")
    print(f"Output shape (QHWD): {output.shape}")
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    print(f"Quaternion norms: {torch.norm(output, p=2, dim=0).mean().item():.4f}")
    
    # Test with tensor
    print("\nTesting with Tensor:")
    tensor_image = torch.randn(3, 100, 100)  # CHW format
    output = transform(tensor_image)
    print(f"Input shape (CHW): {tensor_image.shape}")
    print(f"Output shape (QHWD): {output.shape}")
    
    # Verify quaternion properties
    print("\nVerifying quaternion properties:")
    q = output.view(4, -1)  # Reshape to (4, H*W)
    norms = torch.norm(q, p=2, dim=0)
    print(f"Mean quaternion norm: {norms.mean().item():.4f}")
    print(f"Norm std deviation: {norms.std().item():.4f}")
    print(f"Real component range: [{output[3].min().item():.4f}, {output[3].max().item():.4f}]")

if __name__ == "__main__":
    test_transform()