"""Some common values to use for YOLO"""

import pytest
from torch import Tensor, randn

from object_detection.yolo.v1.settings import IMAGE_SIZE


@pytest.fixture(scope="function")
def x_cnn() -> Tensor:
    return randn(size=(10, 3, IMAGE_SIZE, IMAGE_SIZE))
