import os
import torch

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Version (PyTorch):", torch.version.cuda)
print("Is CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
