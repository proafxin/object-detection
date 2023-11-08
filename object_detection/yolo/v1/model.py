"""Create the YOLOv1 model backbone using units."""

from object_detection.yolo.config import Configuration
from object_detection.yolo.v1.units import unit1, unit2

configuration = Configuration(unit_configs=[])
configuration.add_unit(unit_config=unit1.config)
configuration.add_unit(unit_config=unit2.config)
