"""Test YOLOv1 architecture"""

from torch import Size, Tensor
from torch.nn import Module

from object_detection.yolo.v1.model import YOLOv1


def test_yolov1_architecture(x_cnn: Tensor) -> None:
    yolo = YOLOv1()
    assert isinstance(yolo, Module)

    out = yolo(x_cnn)

    assert isinstance(out, Tensor)
    assert out.shape == Size([10, 1470])
