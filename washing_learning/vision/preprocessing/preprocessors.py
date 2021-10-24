"""
Every used image preprocessor will be found below.
They all implements a :meth:`preprocess(self, image) <preprocess>` method and can be used as follows:

Example:
    >>> sp = SimplePreprocessor(128, 128)
    >>> ss = SimpleScaler()
"""
# Third-party libraries
import cv2
import imutils
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.preprocessing.image import img_to_array

__all__ = ["SimplePreprocessor", "SimpleScaler"]


class SimplePreprocessor:
    """
    A simple preprocessor used to resize a given set of images. This class is `washing_learning.vision.datasets.dataloaders
    compatible.

    Args:
        width (float) : The image width after being processed
        height (float) : The image height after being processed

    """

    def __init__(self, width: float, height: float, inter=cv2.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class SimpleScaler:
    """
    A simple preprocessors that scale every pixels in range [0, 1] (if `factor` = 255.)

    Args:
        factor (float) : The factor used to resize every pixel of the images.

    """

    def __init__(self, factor: float = 255.0) -> None:
        self.factor = factor

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image / self.factor
