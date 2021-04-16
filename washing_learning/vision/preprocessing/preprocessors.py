# Third-party libraries
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import imutils


class SimplePreprocessor:
    def __init__(self, width: float, height: float, inter=cv2.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class SimpleScaler:
    def __init__(self, factor: float = 255.0) -> None:
        self.factor = factor

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image / self.factor


class PatchPreprocessor:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]


class MeanPreprocessor:
    def __init__(self, r_mean: float, g_mean: float, b_mean: float) -> None:
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        return cv2.merge([B, G, R])


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None) -> None:
        self.data_format = data_format

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return img_to_array(image, data_format=self.data_format)


class CropPreprocessor:
    def __init__(self, width: float, height: float, horiz: bool = True, inter=cv2.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        crops = []
        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h],
        ]

        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            # compute the horizontal mirror flips for each crop
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)
        # return the set of crops
        return np.array(crops)


class AspectAwarePreprocessor:
    def __init__(self, width: float, height: float, inter=cv2.INTER_AREA) -> None:
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
        (h, w) = image.shape[:2]
        image = image[dH : h - dH, dW : w - dW]
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
