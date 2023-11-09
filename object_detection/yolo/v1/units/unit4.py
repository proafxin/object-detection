"""4th unit of YOLOv1 model."""

from object_detection.yolo.config import (
    BlockConfiguration,
    Convolution2dConfiguration,
    LayerConfiguration,
    MaxPool2dConfiguration,
)

block1_cell1 = Convolution2dConfiguration(
    kernel_size=1, stride=1, padding=0, in_channels=512, out_channels=256
)
block1_cell2 = Convolution2dConfiguration(
    kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512
)
block1 = BlockConfiguration(block=[])
block1.add_single_cells(block1_cell1, block1_cell2)

block2_cell1 = Convolution2dConfiguration(
    kernel_size=1, stride=1, padding=0, in_channels=512, out_channels=512
)
block2_cell2 = Convolution2dConfiguration(
    kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=1024
)
block2 = BlockConfiguration(block=[])
block2.add_single_cells(block2_cell1, block2_cell2)

block3_cell1 = MaxPool2dConfiguration()
block3 = BlockConfiguration(block=[])
block3.add_single_cells(block3_cell1)

layer = LayerConfiguration(layer=[(block1, 4), (block2, 1), (block3, 1)])

config = layer
