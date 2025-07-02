"""
File: utils/training.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module contains functions for training, validating, and testing a PyTorch model.
It includes functions for training the model for one epoch, validating the model,
and testing the model on a test set.

Functions:
---------
- train: Trains the model for one epoch and returns the average loss and accuracy.
- validation: Validates the model on the validation set and saves the model weights
    if the validation accuracy is higher than the previous best accuracy.
- test: Tests the model on the test set and returns the average accuracy.

Requirements:
------------
- numpy: For numerical operations.
- torch: For building and training the model.
"""

# Import built-in libraries
from typing import Literal, Tuple

# Import third-party libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
from constants import WEIGHT_PATH


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    accuracy: nn.Module,
    dataloader: DataLoader,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Parameters:
        - model (nn.Module): The model to train.
        - optimizer (torch.optim.Optimizer): The optimizer for training.
        - criterion (nn.Module): The loss function.
        - accuracy (nn.Module): The accuracy metric.
        - dataloader (DataLoader): The DataLoader for the training data.
        - device (Literal["cpu", "cuda"]): The string "cpu" or "cuda" indicating the
          device for model computation.

    Returns:
    -------
        - Tuple[float, float]: The average loss and accuracy for the epoch.
    """
    model.train()
    loss_lst = []
    acc_lst = []
    for batch_data in dataloader:
        inputs, labels = batch_data  # inputs: torch.Tensor, labels: torch.Tensor
        inputs, labels = inputs.to(device), labels.to(device)

        logits: torch.Tensor = model(inputs).squeeze()
        preds_probs: torch.Tensor = torch.softmax(logits, dim=1)
        preds: torch.Tensor = torch.argmax(preds_probs, dim=1)

        loss: torch.Tensor = criterion(logits, labels)
        acc: float = accuracy(labels, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append the loss and accuracy to the lists
        loss_lst.append(loss.item())
        acc_lst.append(acc.item())
    return np.mean(loss_lst), np.mean(acc_lst)


def validation(
    model: nn.Module,
    criterion: nn.Module,
    accuracy: nn.Module,
    dataloader: DataLoader,
    epoch_num: int,
    device: Literal["cpu", "cuda"] = "cpu",
) -> Tuple[float, float, float, float]:
    """
    Validate the model on the validation set. This function evaluates the model
    on the validation set and saves the model weights if the validation accuracy
    is higher than the previous best accuracy.

    Parameters:
    ----------
        - model (nn.Module): The model to validate.
        - criterion (nn.Module): The loss function.
        - accuracy (nn.Module): The accuracy metric.
        - dataloader (DataLoader): The DataLoader for the validation data.
        - epoch_num (int): The current epoch number.
        - device (Literal["cpu", "cuda"]): The string "cpu" or "cuda" indicating the
          device for model computation.

    Returns:
    -------
        - Tuple[float, float, float, float]: The average loss, accuracy,
          highest accuracy, and lowest loss for the epoch.
    """

    model.eval()  # Set the model to evaluation mode

    # Initialize variables to track the best accuracy and loss
    loss_lst = []
    acc_lst = []

    with torch.inference_mode():  # Disable gradient calculation for validation
        for batch_data in dataloader:
            inputs, labels = batch_data  # inputs: torch.Tensor, labels: torch.Tensor
            inputs, labels = inputs.to(device), labels.to(device)

            logits: torch.Tensor = model(inputs).squeeze()
            preds_probs: torch.Tensor = torch.softmax(logits, dim=1)
            preds: torch.Tensor = torch.argmax(preds_probs, dim=1)

            loss: torch.Tensor = criterion(logits, labels)
            acc: float = accuracy(labels, preds)

            loss_lst.append(loss.item())
            acc_lst.append(acc.item())

        if epoch_num == 0:
            highest_acc: float = np.mean(acc_lst)
            lowest_loss: float = np.mean(loss_lst)
        elif highest_acc < np.mean(acc_lst):
            highest_acc: float = np.mean(acc_lst)
            lowest_loss: float = np.mean(loss_lst)
            torch.save(model.state_dict(), WEIGHT_PATH)

    return np.mean(loss_lst), np.mean(acc_lst), highest_acc, lowest_loss


def test(
    model: nn.Module,
    dataloader: DataLoader,
    accuracy: nn.Module,
    device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    """
    Test the model on the test set.
    This function evaluates the model on the test set and returns the average
    accuracy.

    Parameters:
    ----------
        - model (nn.Module): The model to test.
        - dataloader (DataLoader): The DataLoader for the test data.
        - accuracy (nn.Module): The accuracy metric.
        - device (Literal["cpu", "cuda"]): The string "cpu" or "cuda" indicating the
          device for model computation.

    Returns:
    -------
        - float: The average accuracy of the model on the test set.
    """
    model.eval()
    acc_lst = []
    with torch.inference_mode():
        for batch_data in dataloader:
            inputs, labels = batch_data  # inputs: torch.Tensor, labels: torch.Tensor
            inputs, labels = inputs.to(device), labels.to(device)

            logits: torch.Tensor = model(inputs).squeeze()
            preds_probs: torch.Tensor = torch.softmax(logits, dim=1)
            preds: torch.Tensor = torch.argmax(preds_probs, dim=1)

            acc: float = accuracy(labels, preds)

            acc_lst.append(acc.item())
    return np.mean(acc_lst)
