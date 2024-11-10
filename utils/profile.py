# utils/profile.py

from thop import profile
from thop import clever_format
import torch 

def get_model_complexity(model, input_size):
    """
    Calculate the FLOPs and number of parameters of the model.

    Args:
        model (nn.Module): The PyTorch model.
        input_size (tuple): The size of the input tensor (C, H, W).

    Returns:
        str: Formatted string of FLOPs and parameters.
    """
    macs, params = profile(model, inputs=(torch.randn(1, *input_size),), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    return f"FLOPs: {macs}, Params: {params}"
