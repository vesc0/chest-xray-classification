"""
Accelerator selection and synchronization.

Split out of train.py so that evaluation, explainability, and the CLI can pick a
device without importing the training stack. Depends on nothing but torch, which
keeps it at the bottom of the import graph alongside config.
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

    CUDA and MPS dispatch kernels asynchronously, so a timer stopped without
    this measures how long it took to *queue* the work, not to run it —
    understating GPU time several-fold. Required around any timed region.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
