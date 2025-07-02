"""
File: explainability.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
date: 2023-10-15

Description:
-----------
This script is used to perform explainability analysis on a pre-trained Vision
Transformer (ViT) model for autism detection. It loads the model, reads an image,
performs predictions, and generates explainability maps using LIME and RISE.

Functions:
----------
- main: The main function that orchestrates the explainability process, including
  loading the model, reading the image, generating explainability scores, and plotting
  the results.

Requirements:
------------
- numpy: For numerical operations.
- torch: For model loading and tensor operations.
"""

# Import built-in libraries
import os
from typing import Literal

# Import third-party libraries
import numpy as np
import torch

# Import local modules
from config import CLASS_NUM
from constants import ROOT_PATH, SEED_NUMBER, WEIGHT_PATH
from utils import (
    load_model,
    read_img,
    get_lime,
    get_rise,
    explainability_metric,
    plot_explainabilty,
    set_seed_and_get_device,
)


def main() -> None:
    """
    Main function for explainability. It loads a pre-trained model, reads an image,
    performs predictions, and generates explainability scores using LIME and RISE. It also
    computes an explainability metric and plots the results.
    Parameters:
    ----------
        - None
    Returns:
    -------
        - None
    """
    # Setting up the environment
    device: Literal["cpu", "cuda"] = set_seed_and_get_device(seed_num=SEED_NUMBER)
    file_name: str = "001.jpg"
    type_name: Literal["autistic", "non-autistic"] = "autistic"
    file_path: str = os.path.join(ROOT_PATH, type_name, file_name)
    label: torch.Tensor = torch.ones(
        1, dtype=torch.int8
    )  # 1 for autistic, 0 for non-autistic

    # Loading the pre-trained model
    model, auto_transform = load_model(
        class_num=CLASS_NUM
    )  # model: nn.Module, auto_transform: transforms.Compose
    model = model.to(device)
    model.load_state_dict(torch.load(WEIGHT_PATH))

    # Reading the image and making predictions
    img: torch.Tensor = read_img(file_path)

    model.eval()
    logits = model(auto_transform(img).unsqueeze(0).to(device))
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    print(f"Predicted label: {preds.item()}, Actual label: {label.item()}")

    # Explainability Part
    # Lime Explanation
    lime_score: np.ndarray = get_lime(
        image=img, model=model, transform=auto_transform, device=device
    )

    # Rise Explanation
    rise_score: np.ndarray = get_rise(
        image=img, model=model, transform=auto_transform, device=device
    )

    # Explainability Metric
    mode: Literal["del", "ins"] = "del"
    lime_metric_result, rise_metric_result = explainability_metric(
        model=model,
        image=img,
        transform=auto_transform,
        lime_score=lime_score,
        rise_score=rise_score,
        mode=mode,
        device=device,
    )  # lime_metric_result: np.ndarray, rise_metric_result: np.ndarray

    # Plotting the Metric Results
    plot_explainabilty(y_lime=lime_metric_result, y_rise=rise_metric_result, mode=mode)


if __name__ == "__main__":
    main()
