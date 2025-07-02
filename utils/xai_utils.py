"""
File: utils/xai_utils.py
Author: Ali Reza (Aro) Omrani
Email: omrani.alireza95@gmail.com
Date: 2023-10-15

Description:
-----------
This module provides utility functions for explainability in image classification tasks.
It includes functions to create a new model wrapper, classify images, generate LIME and
RISE explanations, apply Gaussian blur, and compute explainability metrics. It also
includes a function to plot the explainability results.

Class:
-------
- New_Model: A wrapper class for a neural network model that handles input size and
  device placement. This class is specifically designed to work with the RISE explainer,
    which requires a specific input size and device for the model.
    - predict: A method that takes an input array X, processes it, and returns the model's
      predictions.

Functions:
---------
- classifier_fn: Classifies input images using a specified model and device.
- get_lime: Generates LIME explanations for a given image using a specified model and
    transformation.
- get_rise: Generates RISE explanations for a given image using a specified model and
    transformation.
- gkern: Creates a Gaussian kernel of specified size and standard deviation.
- blur: Applies Gaussian blur to an input image using a Gaussian kernel.
- explainability_metric: Computes the explainability metric for a given image and model
    using LIME and RISE scores.
- plot_explainability: Plots the explainability metrics for LIME and RISE.

Requirements:
------------
- numpy: For numerical operations and array manipulations.
- matplotlib: For plotting the results.
- lime: For generating LIME explanations.
- scipy: For applying Gaussian filters.
- torch: For building and evaluating the model.
- torchvision: For image transformations and preprocessing.
- xailib: For explainability metrics and explainers.

Note:
-----
The `lime` library is imported mainly for documentation purposes in this code.
You can remove it from the imports if you don’t use it directly. However, since it
serves as the background for explainability metrics, it must be installed to
avoid runtime errors.

To install the XAI-Library, run the following command in your terminal:

    pip install XAI-Library
"""

# Import built-in libraries
from typing import Dict, Literal

# Import third-party libraries
import numpy as np
from lime.lime_image import ImageExplanation
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc
import torch, torch.nn as nn
import torchvision.transforms as transforms
from xailib.explainers.lime_explainer import LimeXAIImageExplainer
from xailib.explainers.rise_explainer import RiseXAIImageExplainer
from xailib.metrics.insertiondeletion import ImageInsDel


class New_Model:
    """
    Wrapper class for the model to handle input size and device placement. This class is
    defined to be used for the RISE explainer, which requires a specific input size and
    device for the model.

    Attributes:
    ----------
        - model (nn.Module): The neural network model to be used for predictions.
        - input_size (tuple[int, int]): The expected input size of the model.
        - device (Literal["cpu", "cuda"]): The device on which the model is placed.

    Methods:
    -------
        - predict(X: np.ndarray) -> np.ndarray:
            Takes an input array X, processes it, and returns the model's predictions.


    Example:
    >>> model = New_Model(nn_model, (224, 224), "cuda")
    >>> predictions = model.predict(input_array)

    Note:
    -----
    This class is specifically designed to work with the RISE explainer and may not be
    suitable for other types of explainers or models that do not require a specific input
    size or device. It is important to ensure that the model passed to this class is
    compatible with the expected input size and device.
    """

    def __init__(
        self, bb: nn.Module, input_size: tuple[int, int], device: Literal["cpu", "cuda"]
    ) -> None:
        """
        Initializes the New_Model class with a given model, input size, and device.

        Parameters:
        ----------
            - bb (nn.Module): The neural network model to be wrapped.
            - input_size (tuple[int, int]): The expected input size of the model.
            - device (Literal["cpu", "cuda"]): The device on which the model is placed.

        Returns:
        -------
            - None: The constructor does not return any value.
        """
        self.model: nn.Module = bb
        self.input_size: tuple[int, int] = input_size
        self.device: Literal["cpu", "cuda"] = device

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the input array X using the wrapped model.

        Parameters:
        ----------
            - X (np.ndarray): The input array for which predictions are to be made.

        Returns:
        -------
            - np.ndarray: The model's predictions for the input array X.
        """
        with torch.no_grad():
            images: torch.Tensor = torch.tensor(X).permute(0, 3, 1, 2).float()
            return (
                torch.nn.functional.softmax(self.model(images.to(self.device)), dim=1)
                .cpu()
                .detach()
                .numpy()
            )


def classifier_fn(
    images: np.ndarray, model: nn.Module, device: Literal["cpu", "cuda"]
) -> np.ndarray:
    """
    Classifies the input images using the specified model and device. This function is
    designed to be used with the LIME explainer, which requires a function that takes
    an array of images and returns their predicted class probabilities.

    Parameters:
    ----------
        - images (np.ndarray): An array of images to be classified.
        - model (nn.Module): The neural network model used for classification.
        - device (Literal["cpu", "cuda"]): The device on which the model is placed.

    Returns:
    -------
        - np.ndarray: An array of predicted class probabilities for the input images.

    Usage:
    -----
    >>> predictions = classifier_fn(images, model, device)

    Note:
    -----
    This function is designed to be used with the LIME explainer, which requires a function that takes
    an array of images and returns their predicted class probabilities.
    """
    with torch.no_grad():
        images: torch.Tensor = torch.tensor(images).permute(0, 3, 1, 2)
        pred: np.ndarray = (
            torch.nn.functional.sigmoid(model(images.to(device))).detach().cpu().numpy()
        )
        return pred


def get_lime(
    image: torch.Tensor,
    model: nn.Module,
    transform: transforms.Compose,
    device: Literal["cpu", "cuda"],
    num_samples: int = 5000,
) -> np.ndarray:
    """
    Generates LIME explanations for the given image using the specified model and
    transformation. This function is designed to explain the predictions of a model on a
    single image.

    Parameters:
    ----------
        - image (torch.Tensor): The input image for which LIME explanations are to be generated.
        - model (nn.Module): The neural network model used for generating explanations.
        - transform (transforms.Compose): The transformation applied to the image before passing it to the model.
        - device (Literal["cpu", "cuda"]): The device on which the model is placed.
        - num_samples (int): The number of samples to use for generating LIME explanations.

    Returns:
    -------
        - np.ndarray: An array of LIME scores for the input image, representing the
          importance of each pixel in the image for the model's prediction.

    Usage:
    -----
    >>> lime_scores = get_lime(image, model, transform, device, num_samples=5000)

    Note:
    -----
    This function is designed to be used with the LIME explainer, which requires a function that takes
    an array of images and returns their predicted class probabilities.
    """
    # This transform is used for displaying the picture
    display_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )

    # Create the Explainer
    lm: LimeXAIImageExplainer = LimeXAIImageExplainer(model)

    # Fit the Explainer
    lm.fit()

    # Explain an Instance
    explanation: ImageExplanation = lm.explain(
        transform(image).permute(1, 2, 0).numpy(),
        lambda x: classifier_fn(x, model, device),
        num_samples=num_samples,
    )

    # Plot the results
    lm.plot_lime_values(display_transform(image).permute(1, 2, 0).numpy(), explanation)
    ind: int = explanation.top_labels[0]
    dict_heatmap: Dict[int, float] = dict(explanation.local_exp[ind])
    lime_score: np.ndarray = np.vectorize(dict_heatmap.get)(explanation.segments)

    return lime_score


def get_rise(
    image: torch.Tensor,
    model: nn.Module,
    transform: transforms.Compose,
    device: Literal["cpu", "cuda"],
) -> np.ndarray:
    """
    Generates RISE explanations for the given image using the specified model and
    transformation. This function is designed to explain the predictions of a model on a
    single image.

    Parameters:
    ----------
        - image (torch.Tensor): The input image for which RISE explanations are to be generated.
        - model (nn.Module): The neural network model used for generating explanations.
        - transform (transforms.Compose): The transformation applied to the image before passing it to the model.
        - device (Literal["cpu", "cuda"]): The device on which the model is placed.

    Returns:
    -------
        - np.ndarray: An array of RISE scores for the input image, representing the
          importance of each pixel in the image for the model's prediction.

    Usage:
    -----
    >>> rise_scores = get_rise(image, model, transform, device)

    Note:
    -----
    This function is designed to be used with the RISE explainer, which requires a function that takes
    an array of images and returns their predicted class probabilities.
    """
    # This transform is used for displaying the picture
    display_transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )

    model_new: New_Model = New_Model(model, (224, 224), device)
    rise: RiseXAIImageExplainer = RiseXAIImageExplainer(model_new)

    N: int = 1000  # number of random masks
    s: int = 10  # cell_size = input_shape / s
    p1: float = 0.9  # masking probability

    rise.fit(N, s, p1)

    explanation: np.ndarray = rise.explain(transform(image).permute(1, 2, 0).numpy())
    rise_score: np.ndarray = explanation[0, :]
    _, ax = plt.subplots(1, 3, figsize=(10, 5))  # ax: plt.Axes

    # Plot the Explanation results
    ax[0].set_title("Original Image")
    ax[0].imshow(display_transform(image).permute(1, 2, 0).numpy(), cmap="gray")
    ax[0].axis("off")

    ax[1].set_title("RISE Explanation")
    ax[1].imshow(explanation[0, :], cmap="jet")
    ax[1].axis("off")

    ax[2].set_title("Overlay Explanation")
    ax[2].imshow(display_transform(image).permute(1, 2, 0).numpy(), cmap="gray")
    ax[2].imshow(explanation[0, :], cmap="jet", alpha=0.5)
    ax[2].axis("off")
    return rise_score


def gkern(klen: int, nsig: int) -> torch.Tensor:
    """
    Function that creates a Gaussian kernel of size klen x klen with standard deviation
    nsig. This kernel can be used for blurring images.

    Parameters:
    ----------
        - klen (int): The length of the kernel (both width and height).
        - nsig (int): The standard deviation of the Gaussian distribution.

    Returns:
    -------
        - torch.Tensor: A tensor representing the Gaussian kernel of shape (3, 3, klen, klen).

    Usage:
    -----
    >>> kernel = gkern(klen=25, nsig=25)

    Note:
    -----
    This function is designed to create a Gaussian kernel for image processing for
    Insertion and Deletion explainability metrics. The kernel is created by first
    generating a Dirac delta function in the
    """
    CH: int = 3  # number of channels, e.g., RGB

    # create nxn zeros
    inp: np.ndarray = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen // 2, klen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k: np.ndarray = gaussian_filter(inp, nsig)
    kern: np.ndarray = np.zeros((CH, CH, klen, klen))
    for i in range(CH):
        kern[i, i] = k
    return torch.from_numpy(kern.astype("float32"))


def blur(image: torch.Tensor, klen: int = 25, ksig: int = 25) -> torch.Tensor:
    """
    Applies a Gaussian blur to the input image using a Gaussian kernel of size klen x
    klen with standard deviation ksig. This function is designed to be used for the
    Insertion and Deletion explainability metrics.

    Parameters:
    ----------
        - image (torch.Tensor): The input image to be blurred.
        - klen (int): The length of the kernel (both width and height).
        - ksig (int): The standard deviation of the Gaussian distribution.

    Returns:
    -------
        - torch.Tensor: The blurred image as a tensor.

    Usage:
    -----
    >>> blurred_image = blur(image, klen=25, ksig=25)

    Note:
    -----
    This function is designed to create a Gaussian kernel for image processing for
    Insertion and Deletion explainability metrics. The kernel is created by first
    generating a Dirac delta function in the center of the kernel, then applying a
    Gaussian filter to smooth it, resulting in a Gaussian filter mask. The image is then
    convolved with this kernel to produce the blurred output.
    """
    kern: torch.Tensor = gkern(klen, ksig)
    image: torch.Tensor = torch.tensor(image).float()
    return nn.functional.conv2d(image, kern, padding=klen // 2)


def explainability_metric(
    model: nn.Module,
    image: torch.Tensor,
    transform: transforms.Compose,
    lime_score: np.ndarray,
    rise_score: np.ndarray,
    mode: Literal["del", "ins"],
    device: Literal["cpu", "cuda"],
    step: int = 224,
):
    """
    Computes the explainability metric for a given image and model. This function
    generates predictions using the model, applies the LIME and RISE scores, and then
    computes the final explainability metric.

    Parameters:
    ----------
        - model (nn.Module): The model to be used for predictions.
        - image (torch.Tensor): The input image for which the explainability metric is
          computed.
        - transform (transforms.Compose): The transformation to be applied to the image.
        - lime_score (np.ndarray): The LIME score for the image.
        - rise_score (np.ndarray): The RISE score for the image.
        - mode (Literal["del", "ins"]): The mode of the explainability metric (deletion
          or insertion).
        - device (Literal["cpu", "cuda"]): The device to run the model on.
        - step (int): The step size for the metric computation.

    Returns:
    -------
        - tuple[np.ndarray, np.ndarray]: A tuple containing the LIME and RISE
          explainability metrics for the input image.

    Usage:
    -----
    >>> lime_result, rise_result = explainability_metric(
    ...     model=model,
    ...     image=image,
    ...     transform=transform,
    ...     lime_score=lime_score,
    ...     rise_score=rise_score,
    ...     mode="del",
    ...     device=device,
    ...     step=224
    ... )

    Note:
    -----
    This function is designed to compute the explainability metric for a given image and
    model. It leverages the LIME and RISE scores to provide insights into the model's
    predictions.

    """

    def predict(image: np.ndarray) -> torch.Tensor:
        """
        Predict the class probabilities for a given image using the model.

        Parameters:
        ----------
            - image (np.ndarray): The input image to be classified.

        Returns:
        -------
            - torch.Tensor: The predicted class probabilities for the input image.
        Usage:
        -----
            >>> pred = predict(image)
        Note:
        -----
        This function is designed to be used with the RISE explainer, which requires a
        function that takes an array of images and returns their predicted class
        probabilities.
        """
        with torch.no_grad():
            pred: torch.Tensor = torch.nn.functional.softmax(
                model(torch.from_numpy(image).to(device)), dim=1
            )
            return pred

    if mode == "ins":
        metric: ImageInsDel = ImageInsDel(
            predict=predict, mode=mode, step=step, substrate_fn=torch.zeros_like
        )
    else:
        metric: ImageInsDel = ImageInsDel(
            predict=predict, mode=mode, step=step, substrate_fn=blur
        )

    rise_result: np.ndarray = metric(
        transform(image).unsqueeze(0).numpy(), step, rise_score, rgb=True
    )
    lime_result: np.ndarray = metric(
        transform(image).unsqueeze(0).numpy(), step, lime_score, rgb=True
    )

    return lime_result, rise_result


def plot_explainabilty(
    y_lime: np.ndarray, y_rise: np.ndarray, mode: Literal["del", "ins"], step: int = 224
) -> None:
    """
    Plots the explainability metrics for LIME and RISE. This function generates a plot
    showing the accuracy of the model as a function of the percentage of pixels removed
    or inserted, depending on the mode specified.

    Parameters:
    ----------
        - y_lime (np.ndarray): The LIME explainability scores.
        - y_rise (np.ndarray): The RISE explainability scores.
        - mode (Literal["del", "ins"]): The mode of the explainability metric (deletion
          or insertion).
        - step (int): The step size for the metric computation.

    Returns:
    -------
        - None

    Usage:
    >>> plot_explainabilty(y_lime, y_rise, mode="del", step=224)

    Note:
    -----
    This function is designed to visualize the explainability metrics for LIME and RISE.
    """
    x: np.ndarray = np.arange(len(y_lime)) / (224 * 224) * step
    x[-1] = 1.0

    for name, y in zip(["lime", "rise"], [y_lime, y_rise]):
        plt.plot(x, y, label=f"{name}: {np.round(auc(x, y),4)}")
        plt.fill_between(x, y, alpha=0.4)
    if mode == "del":
        plt.xlabel("Percentage of pixel removed", fontsize=20)
    else:
        plt.xlabel("Percentage of pixel inserted", fontsize=20)
        plt.ylabel("Accuracy of the model", fontsize=20)
        plt.legend(loc="lower left", fontsize=20)
