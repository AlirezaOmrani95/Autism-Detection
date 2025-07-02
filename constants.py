"""
File: constants.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This file is used to define constants used throughout the autism detection project,
such as seed number, root path for datasets, and the path to the pre-trained model
weights. These constants are used in various parts of the project to ensure
consistency and ease of configuration.

Variables:
---------
    - SEED_NUMBER: int
        The seed number for random number generation to ensure reproducibility.
    - ROOT_PATH: str
        The root path where the dataset is stored.
    - WEIGHT_PATH: str
        The path to the pre-trained model weights.
"""

SEED_NUMBER: int = (
    1  # Seed number for random number generation to ensure reproducibility
)

ROOT_PATH: str = "./dataset"  # Root path where the dataset is stored

WEIGHT_PATH: str = "./weight/best_model.pth"  # Path to the pre-trained model weights
