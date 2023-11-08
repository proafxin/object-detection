"""Some common values to use for YOLO"""

import pytest
from torch import Tensor, randn


@pytest.fixture(scope="function")
def x_cnn() -> Tensor:
    return randn(size=(10, 3, 448, 448))
