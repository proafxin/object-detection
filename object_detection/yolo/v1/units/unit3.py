"""Second unit of YOLOv1."""

from object_detection.yolo.config import LayerConfiguration, MaxPool2dConfiguration
from object_detection.yolo.v1.units.common import convolution_block

block1 = convolution_block(
    kernel_size=1, stride=1, padding=0, in_channels=192, out_channels=128
)
block2 = convolution_block(
    kernel_size=3, stride=1, padding=1, in_channels=128, out_channels=256
)
block3 = convolution_block(
    kernel_size=1, stride=1, padding=0, in_channels=256, out_channels=256
)
block4 = convolution_block(
    kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512
)

layer1 = LayerConfiguration(layer=[])
layer1.add_blocks(block_config=block1)
layer1.add_blocks(block_config=block2)
layer1.add_blocks(block_config=block3)
layer1.add_blocks(block_config=block4)

layer1.add_single_cell(cell_config=MaxPool2dConfiguration())

config = layer1
