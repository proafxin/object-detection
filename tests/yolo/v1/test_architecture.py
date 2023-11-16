"""Test YOLOv1 architecture"""

from torch import Size, Tensor
from torch.nn import Module

from object_detection.yolo.v1.hyperparams import hyperparams
from object_detection.yolo.v1.model import YOLOv1


def test_yolov1_architecture(batch_size: int, x_cnn: Tensor) -> None:
    yolo = YOLOv1()
    assert isinstance(yolo, Module)

    out = yolo(x_cnn)

    assert isinstance(out, Tensor)
    assert out.shape == Size(
        [
            batch_size,
            (hyperparams.num_boxes * 5 + hyperparams.num_class)
            * hyperparams.grid_size
            * hyperparams.grid_size,
        ]
    )
