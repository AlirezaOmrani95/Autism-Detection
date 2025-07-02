"""
File: utils/__init__.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module provides utility functions for data loading, model handling, training,
validation, testing, and explainability in the context of autism detection using
Vision Transformers (ViT). It includes functions for reading images, loading
dataloaders, training and validating models, testing models, and generating
explanations using LIME and RISE methods.

Functions:
-----------
- read_img: Reads an image from a given path.
- load_dataloader: Loads a DataLoader for the dataset.
- set_seed_and_get_device: Sets the random seed and returns the device
  (CPU or GPU).
- load_model: Loads a pre-trained Vision Transformer model.
- plot_history: Plots the training history of the model.
- write_history_to_csv: Writes the training history to a CSV file.
- get_lime: Generates explanations using the LIME method.
- get_rise: Generates explanations using the RISE method.
- explainability_metric: Computes the explainability metric for the model.
- plot_explainabilty: Plots the explainability results.
- train: Trains the model for one epoch.
- validation: Validates the model on the validation set.
- test: Tests the model on the test set.
- get_insertion_deletion: Computes insertion and deletion metrics for
  explainability.
"""

from data_utils import (
    read_img,
    load_dataloader,
)
from general import set_seed_and_get_device
from model_utils import (
    plot_history,
    load_model,
    write_history_to_csv,
)
from xai_utils import (
    get_lime,
    get_rise,
    explainability_metric,
    plot_explainabilty,
)

from training import train, validation, test

__all__ = [
    "explainability_metric",
    "get_insertion_deletion",
    "get_lime",
    "get_rise",
    "load_dataloader",
    "load_model",
    "plot_history",
    "plot_explainabilty",
    "read_img",
    "set_seed_and_get_device",
    "test",
    "train",
    "validation",
    "write_history_to_csv",
]
