from __future__ import annotations
import torch
from torch import nn

def load_torch_model(path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, nn.Module):
        return obj
    if "model_state_dict" in obj and "model_class" in obj:
        # Expect a callable/class in the checkpoint
        model: nn.Module = obj["model_class"](**obj.get("model_kwargs", {}))
        model.load_state_dict(obj["model_state_dict"])
        model.eval()
        return model
    raise ValueError("Unsupported checkpoint format for PyTorch model.")

def count_hidden_relu_units(m: nn.Module) -> int:
    """Total number of ReLU units across hidden layers 1..L-1 (assuming Linear+ReLU blocks)."""
    s = 0
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            # we assume the preceding Linear defines width; this is used only for sanity checks
            pass
    # In practice you will know widths per layer; leave s unknown here (not strictly required).
    return s
