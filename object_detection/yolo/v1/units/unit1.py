"""First unit of YOLOv1 model"""

from object_detection.yolo.config import (
    BlockConfiguration,
    ConvolutionConfiguration,
    MaxPool2dConfiguration,
)

block1_cell1 = ConvolutionConfiguration(
    kernel_size=7, padding=3, in_channels=3, out_channels=64
)
block1_cell2 = MaxPool2dConfiguration()
block1 = BlockConfiguration(block=[])
block1.add_cells(cell_config=block1_cell1, repeat=1)
block1.add_cells(cell_config=block1_cell2, repeat=1)

config = block1
