# Standard libraries
import os
import datetime
from typing import List, Tuple, Union

# Third-party libraries
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType
import PIL

Image = Union[np.ndarray, torch.Tensor, PIL.Image]


class TensorBoardLogger(object):
    def __init__(self, log_dir: str, log_hist: bool = True) -> None:
        """Create a summary writer logging to log_dir."""
        if log_hist:  # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            )
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag: str, value: float, step: int) -> None:
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(
        self, tag_value_pairs: List[Tuple[str, float]], step: int
    ) -> None:
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def graph_summary(
        self,
        model: torch.nn.Module,
        img: TensorType["batch", "channel", "height", "width"],
    ) -> None:
        """Log a graph model"""
        self.writer.add_graph(model, img)

    def image_summary(self, tag: str, img: Image) -> None:
        """Log an image"""
        self.writer.add_image(tag, img)

    def list_of_images_summary(self, tag_img_pairs: List[Tuple[str, Image]]) -> None:
        """Log images"""
        for tag, img in tag_img_pairs:
            self.writer.add_scalar(tag, img)
