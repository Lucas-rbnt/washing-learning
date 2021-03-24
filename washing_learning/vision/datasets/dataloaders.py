# Standard libraries
from typing import List, Callable, Tuple
import os

# Third-party libraries
import cv2
import numpy as np


class SimpleDatasetLoader:
    def __init__(self, preprocessors: List[Callable] = None) -> None:
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths: List[str], verbose: int = -1) -> Tuple[np.ndarray, np.ndarray]:
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
