"""Second unit of YOLOv1."""

from object_detection.yolo.config import LayerConfiguration, MaxPool2dConfiguration
from object_detection.yolo.v1.units.blocks import convolution_block

block1 = convolution_block(
    kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=192
)


layer1 = LayerConfiguration(layer=[])
layer1.add_blocks(block_config=block1)
layer1.add_single_cell(cell_config=MaxPool2dConfiguration())


config = layer1
