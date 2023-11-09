"""Second unit of YOLOv1."""

from object_detection.yolo.config import (
    BlockConfiguration,
    Convolution2dConfiguration,
    MaxPool2dConfiguration,
)

block1_cell1 = Convolution2dConfiguration(
    kernel_size=1, stride=1, padding=0, in_channels=192, out_channels=128
)
block1_cell2 = Convolution2dConfiguration(
    kernel_size=3, stride=1, padding=1, in_channels=128, out_channels=256
)
block1_cell3 = Convolution2dConfiguration(
    kernel_size=1, stride=1, padding=0, in_channels=256, out_channels=256
)
block1_cell4 = Convolution2dConfiguration(
    kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512
)
block1_cell5 = MaxPool2dConfiguration()
block1 = BlockConfiguration(block=[])
block1.add_single_cells(
    block1_cell1, block1_cell2, block1_cell3, block1_cell4, block1_cell5
)

config = block1
