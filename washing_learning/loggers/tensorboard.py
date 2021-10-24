"""
Implement the TensorBoard logger used to visualize metrics in TensorBoard.
"""

import datetime

# Standard libraries
import os
from typing import List, Tuple, Union

# Third-party libraries
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

__all__ = ["TensorBoardLogger"]

Image = Union[np.ndarray, torch.Tensor]


class TensorBoardLogger(object):
    """
    Implement an API to log desired scalars, images, models and other in TensorBoard

    Args:
        log_dir (path) : path to store logging files
        log_hist (boolean) : allows to create a new folder for each declaration of the class.
    """

    def __init__(self, log_dir: str, log_hist: bool = True) -> None:
        if log_hist:  # Check a new folder for each log should be created
            log_dir = os.path.join(
                log_dir, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            )
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag: str, value: float, step: int) -> None:
        """
        This method allow to log a particular scalar at a specific step into TensorBoard.

        Args:
            tag (str) : The scalar will be stored under this tag on TensorBoard.
            value (float) : The scalar that be logged.
            step (int) : The step associated to the scalar on TensorBoard.
        """
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(
        self, tag_value_pairs: List[Tuple[str, float]], step: int
    ) -> None:
        """
        This method allow to log a several scalar at a specific step into TensorBoard.

        Args:
            tag_value_pairs (list of tuple containint a string and a float) : Each element of this list contains a
            scalar value and the tag under which the scalar will be logged.
            step (int) : The step associated to the scalars on TensorBoard.
        """
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def graph_summary(
        self,
        model: torch.nn.Module,
        img: TensorType["batch", "channel", "height", "width"],  # noqa
    ) -> None:
        """
        This method allow to log a model into TensorBoard

        Args:
            model (nn.Module) : the deep learning model to log.
            img (Tensor) : an image to know the model input shape.
        """
        self.writer.add_graph(model, img)

    def image_summary(self, tag: str, img: Image) -> None:
        """
        This method allow to log an image into TensorBoard.

        Args:
            tag (string) : The scalar will be stored under this tag on TensorBoard.
            img (Union[Tensor, np.ndarray, PIL.Image) : The image to log
        """
        self.writer.add_image(tag, img)

    def list_of_images_summary(self, tag_img_pairs: List[Tuple[str, Image]]) -> None:
        """
        This method allow to log several images into TensorBoard.

        Args:
            tag_img_pairs (list of tuple) : each element of the list must contains an Image and the tag under which the
            image will be logged.
        """
        for tag, img in tag_img_pairs:
            self.writer.add_scalar(tag, img)
