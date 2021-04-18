"""
This module implements all kind of data loaders in the Computer Vision field. This module is tf.keras friendly to fill the
gap in the area of data loaders of these libraries.

All these Data Loaders implements a :meth:`load(self, image_paths) <load>` method.
"""
import os

# Standard libraries
from typing import Callable, List, Tuple

# Third-party libraries
import cv2
import numpy as np


class SimpleDatasetLoader:
    """
    This class allows to chain several processors in order to load data in a specific format.
    The images loaded are loaded as numpy array.

    Args:
        preprocessors (list[Callable]) : The preprocessors that will be used in the right order

    Example:
        >>> sdl = SimpleDatasetLoader(preprocessors=None)
        >>> sdl.load(DATASET_PATH)
    """

    def __init__(self, preprocessors: List[Callable] = None) -> None:
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(
        self, image_paths: List[str], verbose: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        # init the list of features and labels
        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            label = image_path.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

        return (np.array(data), np.array(labels))
