import torch


def get_device():
    """Get GPU acceleration if it is available otherwise choose CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
