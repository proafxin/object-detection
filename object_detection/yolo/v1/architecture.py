"""Specification of model."""

from dataclasses import dataclass, field

from object_detection.yolo.config_parser import ConfigurationParser
from object_detection.yolo.units import Model
from object_detection.yolo.v1.model import configuration


@dataclass
class Architecture:
    parser: ConfigurationParser = field(init=False)

    model: Model = field(init=False)

    def __post_init__(self) -> None:
        self.parser = ConfigurationParser(configuration=configuration)
        self.model = self.parser.get_model()
