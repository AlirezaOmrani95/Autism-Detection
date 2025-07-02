"""
File: utils/data_utils.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module provides utility functions for data loading and preprocessing in the
context of autism detection using Vision Transformers (ViT). It includes functions
for reading images, loading dataloaders, and applying transformations to the images.

Requirements:
------------
- matplotlib: For reading images.
- torch: For building and training the model.
- torchvision: For loading datasets and applying transformations.
"""

# Import third-party libraries
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def read_img(path: str) -> torch.Tensor:
    """
    Reads an image from a file and returns it as a PyTorch tensor.

    Parameters:
    ----------
        - path (str): The file path to the image.
    Returns:
    -------
        - torch.Tensor: The image as a tensor with shape (C, H, W), where C is the number of channels,
          H is the height, and W is the width.
    """
    # read an image for explainability aspects.
    img = torch.from_numpy(plt.imread(path)).permute(2, 0, 1)

    return img


def load_dataloader(
    root: str,
    transform: transforms.Compose,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Loads a DataLoader for the dataset.

    Parameters:
    ----------
        - root (str): The root directory of the dataset.
        - transform (transforms.Compose): The transformations to apply to the images.
        - batch_size (int): The batch size for the DataLoader.
        - shuffle (bool): Whether to shuffle the dataset.
        - num_workers (int): The number of worker processes for data loading.
        - pin_memory (bool): Whether to pin memory for faster data transfer to GPU.
        - drop_last (bool): Whether to drop the last incomplete batch.

    Returns:
    -------
        - DataLoader: The DataLoader for the dataset.
    """

    dataset = datasets.ImageFolder(
        root=root,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataloader
