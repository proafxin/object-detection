"""Specification of model."""

from dataclasses import dataclass, field

from torch import Tensor, flatten
from torch.nn import Module

from object_detection.yolo.config_parser import ConfigurationParser
from object_detection.yolo.units import Model
from object_detection.yolo.v1 import darknet
from object_detection.yolo.v1.units import linear_layer


@dataclass
class YOLOv1(Module):
    darknet: Model = field(init=False)
    fully_connected_layer: Model = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        parser = ConfigurationParser(configuration=darknet.configuration)
        self.darknet = parser.get_model().get_sequential()
        parser = ConfigurationParser(configuration=linear_layer.configuration)
        self.fully_connected_layer = parser.get_model().get_sequential()

    def forward(self, x: Tensor) -> Tensor:
        x = self.darknet(x)
        x = flatten(input=x, start_dim=1)
        return self.fully_connected_layer(x)
