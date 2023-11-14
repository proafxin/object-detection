"""First unit of YOLOv1 model"""

from object_detection.yolo.config import LayerConfiguration, MaxPool2dConfiguration
from object_detection.yolo.v1.hyperparams import hyperparams
from object_detection.yolo.v1.units.blocks import convolution_block

block1 = convolution_block(
    kernel_size=7,
    padding=3,
    in_channels=hyperparams.in_channels,
    out_channels=64,
    stride=2,
)


layer1 = LayerConfiguration(layer=[])
layer1.add_blocks(block_config=block1)
layer1.add_single_cell(cell_config=MaxPool2dConfiguration())
config = layer1
