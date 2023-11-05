"""Base block for different units"""

from dataclasses import dataclass
from enum import Enum, auto

from torch import Tensor
from torch.nn import Conv2d, LeakyReLU, MaxPool2d, Module, ReLU


class UnitType(Enum):
    Convolutional = auto()
    MaxPool2d = auto()
    ReLU = auto()
    LeakyReLU = auto()


UNIT_MAPPING: dict[int, Module] = {
    UnitType.Convolutional.value: Conv2d,
    UnitType.LeakyReLU.value: LeakyReLU,
    UnitType.MaxPool2d.value: MaxPool2d,
    UnitType.ReLU.value: ReLU,
}


class Unit:
    def __new__(cls, unit_type: UnitType, **kwargs) -> Module:
        cls.unit_type = unit_type

        cls.unit = UNIT_MAPPING[unit_type.value](**kwargs)

        return cls.unit

    def __call__(self, x: Tensor) -> Tensor:
        return self(x)


@dataclass
class BaseBlock:
    block: list[tuple[Unit, int]]


@dataclass
class BaseLayer:
    layer: list[tuple[BaseBlock, int]]
