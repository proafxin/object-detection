"""Convolution blocks"""

from torch import Tensor
from torch.nn import Module

from object_detection.yolo.common.blocks.base import Unit, UnitType


class CNNUnit(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        **kwargs
    ) -> None:
        super().__init__()
        unit = Unit(
            unit_type=UnitType.Convolutional,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            **kwargs
        )
        self.__cnn = unit.unit

    def forward(self, x: Tensor) -> Tensor:
        return self.__cnn(x)
