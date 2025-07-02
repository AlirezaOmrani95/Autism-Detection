"""
File: utils/model_utils.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module provides utility functions for handling the Vision Transformer (ViT) model
and its training history in the context of autism detection. It includes functions for
plotting the training history, writing the history to a CSV file, and loading the pre-trained
Vision Transformer model with a modified head for classification tasks.

Requirements:
------------
- matplotlib: For plotting the training history.
- pandas: For writing the training history to a CSV file.
- torch: For building and training the model.
- torchvision: For loading the pre-trained Vision Transformer model and its
  transformations.
"""

# Import built-in libraries
from typing import Dict, List

# Import third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
import torchvision as tv


def plot_history(
    history: Dict[str, List[float]],
    highest_acc: float,
    epoch_num: int,
) -> None:
    """
    Plots the training history of the model.

    Parameters:
    ----------
        - history (Dict[str, List[float]]): A dictionary containing the training and
          validation loss and accuracy.
        - highest_acc (float): The highest accuracy achieved during training.
        - epoch_num (int): The number of epochs the model was trained for.

    Returns:
    -------
        - None: The function saves the plot as a PNG file.
    """
    plt.figure(figsize=(10, 7))
    plt.subplot(121)
    plt.title("loss")
    plt.plot(range(epoch_num), history["train_loss"], color="blue", label="train")
    plt.plot(
        range(epoch_num), history["validation_loss"], color="orange", label="valid"
    )
    plt.legend()
    plt.subplot(122)
    plt.title("accuracy")
    plt.plot(range(epoch_num), history["train_acc"], color="blue", label="train")
    plt.plot(range(epoch_num), history["validation_acc"], color="orange", label="valid")
    plt.plot(range(epoch_num), highest_acc, color="gray", label="highest acc")
    plt.legend()
    plt.savefig("./history_diagram_multi_aug.png")


def write_history_to_csv(
    history: Dict[str, List[float]],
    file_name: str = "history_multi_aug.csv",
    index: bool = False,
) -> None:
    """
    Writes the training history to a CSV file.

    Parameters:
    ----------
        - history (Dict[str, List[float]]): A dictionary containing the training and
          validation loss and accuracy.
        - file_name (str): The name of the CSV file to save the history to.
        - index (bool): Whether to include the index in the CSV file.

    Returns:
    -------
        - None: The function saves the history to a CSV file.
    """
    df: pd.DataFrame = pd.DataFrame(history)
    df.to_csv(file_name, index=index)


def load_model(class_num: int) -> tuple[nn.Module, tv.transforms.Compose]:
    """
    Loads the pre-trained Vision Transformer model and its corresponding image
    transformations.

    Parameters:
    ----------
        - class_num (int): The number of classes for the classification task.

    Returns:
    -------
        - tuple[nn.Module, tv.transforms.Compose]:
            - nn.Module: the pre-trained Vision Transformer model with a modified head
              for the specified number of classes.
            - tv.transforms.Compose: the image transformations.
    """
    model_weight = tv.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
    model: nn.Module = tv.models.vit_h_14(weights=model_weight)
    # Freeze all parameters except the projection layer and the head
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the conv_proj layer to allow fine-tuning
    model.conv_proj.requires_grad = True

    # Modify the head of the model to match the number of classes
    model.heads.head = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(model.heads.head.in_features, class_num),
    )

    auto_transform: tv.transforms.Compose = model_weight.transforms()

    return (model, auto_transform)
