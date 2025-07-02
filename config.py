"""
File: config.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This script contains configuration settings for the autism detection project.
It includes constants for batch size, number of epochs, class numbers, and loss function.

variables:
---------
- BATCH_SIZE: int
    The size of the batch for training and evaluation.
- EPOCHS: int
    The number of epochs for training the model.
- CLASS_NUM: int
    The number of classes in the classification task.
- LOSS_FUNCTION: nn.Module
    The loss function used for training the model.
- ACCURACY: Accuracy
    The accuracy metric used for evaluating the model performance.
"""

# Import third-party libraries
from torch import nn
from torchmetrics import Accuracy

BATCH_SIZE: int = 32  # Size of the batch for training and evaluation
EPOCHS: int = 150  # Number of epochs for training the model

CLASS_NUM: int = 2  # Number of classes in the classification task

LOSS_FUNCTION: nn.Module = (
    nn.CrossEntropyLoss()
)  # Loss function used for training the model
ACCURACY: Accuracy = Accuracy(
    "multiclass", num_classes=CLASS_NUM
)  # Accuracy metric used for evaluating the model performance
