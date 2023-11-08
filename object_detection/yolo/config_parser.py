"""Configuration parser to parse configuration dictionary and define model."""


from dataclasses import dataclass

from object_detection.yolo.units import Unit


@dataclass
class ConfigurationParser:
    config: list[dict]

    def validate(self) -> bool | list[Unit]:
        return False
