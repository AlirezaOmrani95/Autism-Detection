"""
File: train.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This script trains a Vision Transformer (ViT) model on a dataset for autism
detection. It includes functions for training, validation, and saving the best
model weights based on validation accuracy. It also handles data loading, model
initialization, and history tracking for loss and accuracy.

Functions:
---------
- main: The main function that orchestrates the training and validation process.

Requirements:
------------
- torch: For building and training the model.
- tqdm: For progress bars.
"""

# Import built-in libraries
import os
from typing import Dict, Literal

# Import third-party libraries
import torch
from torch import (
    nn,
    optim,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local modules
from utils import (
    load_model,
    plot_history,
    set_seed_and_get_device,
    load_dataloader,
    write_history_to_csv,
    train,
    validation,
)
from constants import (
    SEED_NUMBER,
    ROOT_PATH,
)
from config import (
    BATCH_SIZE,
    EPOCHS,
    CLASS_NUM,
    LOSS_FUNCTION,
    ACCURACY,
)


def main() -> None:
    """
    Main function to train and validate the model. It initializes the model, loads the
    data, sets up the optimizer, and runs the training and validation loops. It also
    tracks the history of loss and accuracy.

    Parameters:
    ----------
        - None

    Returns:
    -------
        - None
    """
    device: Literal["cpu", "cuda"] = set_seed_and_get_device(seed_num=SEED_NUMBER)

    model, auto_transform = load_model(
        class_num=CLASS_NUM
    )  # model: nn.Module, auto_transform: torchvision.transforms.Compose

    train_loader: DataLoader = load_dataloader(
        root=os.path.join(ROOT_PATH, "train"),
        transform=auto_transform,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader: DataLoader = load_dataloader(
        root=os.path.join(ROOT_PATH, "valid"),
        transform=auto_transform,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer: optim.Optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model: nn.Module = model.to(device)
    best_result = {"acc": 0, "loss": 0}

    history: Dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "validation_loss": [],
        "validation_acc": [],
        "highest_acc": [],
    }  # Initialize history dictionary

    for epoch in tqdm(range(EPOCHS)):
        with tqdm() as tepoch:  # Initialize the progress bar for each epoch
            train_loss, train_acc = train(
                model=model,
                optimizer=optimizer,
                criterion=LOSS_FUNCTION,
                accuracy=ACCURACY,
                dataloader=train_loader,
                device=device,
            )  # train_loss: float, train_acc: float

            valid_loss, valid_acc, best_result["acc"], best_result["loss"] = validation(
                model=model,
                criterion=LOSS_FUNCTION,
                accuracy=ACCURACY,
                dataloader=validation_loader,
                epoch_num=epoch,
                device=device,
            )  # valid_loss: float, valid_acc: float, best_result["acc"]: float, best_result["loss"]: float

            tepoch.set_postfix(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "valid_loss": valid_loss,
                    "valid_acc": valid_acc,
                }
            )  # Update the progress bar with current epoch results

            # Append the results to the history, for plotting later
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["validation_loss"].append(valid_loss)
            history["validation_acc"].append(valid_acc)

    plot_history(
        train_loss=history["train_loss"],
        train_acc=history["train_acc"],
        valid_loss=history["validation_loss"],
        valid_acc=history["validation_acc"],
        highest_acc=history["highest_acc"],
        epoch_num=EPOCHS,
    )  # plot the training and validation history
    write_history_to_csv(history=history)  # write the history to a CSV file


if __name__ == "__main__":
    main()
