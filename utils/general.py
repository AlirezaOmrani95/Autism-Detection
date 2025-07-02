"""
File: utils/general.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module provides utility functions for setting random seeds and determining the device
for model computation in the context of autism detection using Vision Transformers (ViT).

Requirements:
------------
- numpy: For setting the random seed for NumPy.
- torch: For setting the random seed for PyTorch and determining the device (CPU or GPU).
"""

# Import built-in libraries
from typing import Literal
import random as rnd

# Import third-party libraries
import numpy as np
import torch


def set_seed_and_get_device(seed_num: int) -> Literal["cpu", "cuda"]:
    """
    Sets seeds for PyTorch, NumPy, and Python's random module for reproducibility, and
    returns the device string for convenience.
    This function ensures reproducible results by setting the random seeds for PyTorch,
    NumPy, and Python's built-in `random` module. It also determines whether to use
    "cuda" or "cpu" for model computation and returns the device string.

    Parameters
    ----------
        - seed_num (int): The seed number to use for all random number generators.

    Returns
    -------
        - device (Literal["cpu", "cuda"]): The string "cpu" or "cuda" indicating the
          device for model computation.

    """
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed_num)  # Set the seed for CPU
    if device == "cuda":
        # Set the seed for all GPUs
        torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)  # Set the seed for NumPy
    rnd.seed(seed_num)  # Set the seed for random

    return device
