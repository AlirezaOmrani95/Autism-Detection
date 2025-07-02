"""
File: test.py
Author: Ali Reza (Aro) Omrani
Email: ali.omrani@example.com
Date: 2023-10-15

Description:
-----------
This script is used to test a pre-trained Vision Transformer (ViT) model on a
dataset for autism detection. It loads the model, applies the necessary
transformations to the input data, and evaluates the model's performance on
the test set.

Functions:
---------
- main: The main function that orchestrates the testing process, including
    loading the model, preparing the test data, and calculating the accuracy.

Requirements:
------------
- torch: For building and evaluating the model.
"""

# Import built-in libraries
import os
from typing import Literal

# Import third-party libraries
import torch, torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
from config import BATCH_SIZE, ACCURACY, CLASS_NUM
from constants import SEED_NUMBER, ROOT_PATH, WEIGHT_PATH
from utils import load_model, set_seed_and_get_device, test, load_dataloader


def main() -> None:
    """
    Main function to run the testing process.
    It initializes the model, loads the test data, and evaluates the model's
    performance on the test set.

    It prints the test accuracy to the console.

    Parameters:
    ----------
        - None

    Returns:
    -------
        - None
    """
    device: Literal["cpu", "cuda"] = set_seed_and_get_device(seed_num=SEED_NUMBER)

    model, auto_transform = load_model(
        CLASS_NUM
    )  # model: nn.Module, auto_transform: torchvision.transforms.Compose

    test_loader: DataLoader = load_dataloader(
        root=os.path.join(ROOT_PATH, "test"),
        transform=auto_transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model: nn.Module = model.to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    test_acc: float = test(
        model=model, dataloader=test_loader, accuracy=ACCURACY, device=device
    )

    print(f"test set:\nacc: {test_acc:0.2f}")  # Print the test accuracy


if __name__ == "__main__":
    main()
