"""Some common values to use for YOLO"""

import pytest
from torch import Tensor, randn

from object_detection.yolo.v1.hyperparams import hyperparams


@pytest.fixture(scope="function")
def x_cnn() -> Tensor:
    return randn(size=(10, 3, hyperparams.image_size, hyperparams.image_size))
