"""Second unit of YOLOv1."""

from object_detection.yolo.config import (
    BlockConfiguration,
    Convolution2dConfiguration,
    MaxPool2dConfiguration,
)

block1_cell1 = Convolution2dConfiguration(
    kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=192
)
block1_cell2 = MaxPool2dConfiguration()
block1 = BlockConfiguration(block=[])
block1.add_cells(cell_config=block1_cell1, repeat=1)
block1.add_cells(cell_config=block1_cell2, repeat=1)

config = block1
