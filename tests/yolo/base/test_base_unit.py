"""A dummy test for initial testing."""


from torch import Size, Tensor
from torch.nn import Conv2d, MaxPool2d, Module

from object_detection.yolo.blocks.base import Unit, UnitType


def test_cnn_unit(x_cnn: Tensor):
    unit = Unit(
        unit_type=UnitType.Convolutional,
        in_channels=3,
        out_channels=8,
        kernel_size=2,
        stride=2,
        padding=0,
        bias=False,
    )

    assert isinstance(unit, Module)
    assert isinstance(unit, Conv2d)

    output = unit(x_cnn)
    assert isinstance(output, Tensor)
    shape = output.shape
    assert isinstance(shape, Size)
    assert shape == Size([10, 8, 5, 5])


def test_maxpool_unit(x_cnn: Tensor):
    unit = Unit(unit_type=UnitType.MaxPool2d, kernel_size=2, stride=2)
    assert isinstance(unit, Module)
    assert isinstance(unit, MaxPool2d)

    output = unit(x_cnn)
    isinstance(output, Tensor)
    shape = output.shape
    assert isinstance(shape, Size)
    assert shape == Size([10, 3, 5, 5])
