"""4th unit of YOLOv1 model."""

from object_detection.yolo.config import LayerConfiguration, MaxPool2dConfiguration
from object_detection.yolo.v1.units.common import convolution_block

conv_block1 = convolution_block(
    kernel_size=1, stride=1, padding=0, in_channels=512, out_channels=256
)
conv_block2 = convolution_block(
    kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512
)

block1 = conv_block1
block1.extend(conv_block2.block)

conv_block3 = convolution_block(
    kernel_size=1, stride=1, padding=0, in_channels=512, out_channels=512
)
conv_block4 = convolution_block(
    kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=1024
)
block2 = conv_block3
block2.extend(conv_block4.block)
layer = LayerConfiguration(layer=[(block1, 4), (block2, 1)])
layer.add_single_cell(cell_config=MaxPool2dConfiguration())

config = layer
