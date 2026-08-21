"""
Accelerator selection and synchronization.

Separate from train.py so evaluation, explainability and the CLI can pick a
device without importing the training stack.
"""

import torch


def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """
    Block until queued accelerator work has finished.

    CUDA and MPS dispatch asynchronously, so a timer stopped without this
    measures queue time, not run time. Required around any timed region.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
