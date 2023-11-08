"""Test YOLOv1 architecture"""

from torch import Size, Tensor
from torch.nn import Module

from object_detection.yolo.v1.architecture import Architecture


def test_yolov1_architecture(x_cnn: Tensor) -> None:
    architecture = Architecture()
    all_cells = architecture.model.all_cells()

    assert isinstance(all_cells, list)
    for cell in all_cells:
        assert isinstance(cell, Module)

    sequential = architecture.model.get_sequential()
    out = sequential(x_cnn)
    assert isinstance(out, Tensor)
    assert out.shape == Size([10, 192, 12, 12])
