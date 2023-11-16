"""Some common values to use for YOLO"""

import pytest
from torch import Tensor, randn

from object_detection.yolo.v1.hyperparams import hyperparams


@pytest.fixture(scope="function")
def batch_size() -> int:
    return 4


@pytest.fixture(scope="function")
def x_cnn(batch_size: int) -> Tensor:
    return randn(
        size=(
            batch_size,
            hyperparams.in_channels,
            hyperparams.image_size,
            hyperparams.image_size,
        )
    )
