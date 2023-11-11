"""Some common functions."""


from object_detection.yolo.config import (
    BatchNorm2dConfiguration,
    BlockConfiguration,
    Convolution2dConfiguration,
    LeakyReLUConfiguration,
)


def convolution_block(
    kernel_size: int, padding: int, in_channels: int, out_channels: int, stride: int
):
    block = BlockConfiguration(block=[])
    cnn = Convolution2dConfiguration(
        kernel_size=kernel_size,
        padding=padding,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    )
    batch_norm = BatchNorm2dConfiguration(num_features=out_channels)
    leaky_relu = LeakyReLUConfiguration()
    block.add_single_cells(cnn, batch_norm, leaky_relu)

    return block
