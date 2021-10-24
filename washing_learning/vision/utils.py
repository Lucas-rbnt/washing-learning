"""
Every utility classes and functions related to vision in Deep Learning are listed below.
"""
# Standard libraries
from typing import Tuple

__all__ = ["compute_padding_conv2d"]


def compute_padding_conv2d(
    mode: str,
    dimensions: Tuple[int, ...],
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...] = (1, 1),
    dilatation: Tuple[int, ...] = (1, 1),
) -> Tuple[int, ...]:

    if mode == "valid":
        return 0, 0
    elif mode == "same":
        return (
            int(
                (
                    (dimensions[0] - 1) * stride[0]
                    - dimensions[0]
                    + dilatation[0] * (kernel[0] - 1)
                    + 1
                )
                / 2
            ),
            int(
                (
                    (dimensions[1] - 1) * stride[1]
                    - dimensions[1]
                    + dilatation[1] * (kernel[1] - 1)
                    + 1
                )
                / 2
            ),
        )
    else:
        raise TypeError(f"{mode} is not existing, please use valid or same instead")
