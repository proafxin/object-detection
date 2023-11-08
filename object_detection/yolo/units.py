"""A single unit of a block."""

from dataclasses import dataclass
from enum import Enum, auto

from torch.nn import Conv2d, LeakyReLU, MaxPool2d, Module, ReLU


class IterableEnum(Enum):
    @classmethod
    def values(cls):
        return [entry.value for entry in cls]


class CellType(Enum):
    Convolutional = auto()
    MaxPool2d = auto()
    ReLU = auto()
    LeakyReLU = auto()


CELL_MAPPING: dict[int, Module] = {
    CellType.Convolutional.value: Conv2d,
    CellType.LeakyReLU.value: LeakyReLU,
    CellType.MaxPool2d.value: MaxPool2d,
    CellType.ReLU.value: ReLU,
}


class MaxPool2dParams(IterableEnum):
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"
    PADDING = "padding"


class ConvolutionParams(IterableEnum):
    KERNEL_SIZE = "kernel_size"
    STRIDE = "stride"
    PADDING = "padding"
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"


class ReLUParams(IterableEnum):
    pass


class LeakyReLUParams(IterableEnum):
    NEGATIVE_SLOPE = "negative_slope"


class UnitType(IterableEnum):
    CELL = auto()
    BLOCK = auto()
    LAYER = auto()


CELL_PARAMS_MAPPING: dict[CellType, IterableEnum] = {
    CellType.Convolutional: ConvolutionParams,
    CellType.LeakyReLU: LeakyReLUParams,
    CellType.MaxPool2d: MaxPool2dParams,
    CellType.ReLU: ReLUParams,
}


class Cell:
    def __new__(cls, cell_type: CellType, **kwargs) -> Module:
        cls.cell_type = cell_type
        param_entries: IterableEnum = CELL_PARAMS_MAPPING[cls.cell_type]
        params = param_entries.values()
        for param in params:
            if param not in kwargs:
                raise KeyError(f"{param} not in {kwargs.keys()}")

        cls.cell = CELL_MAPPING[cell_type.value](**kwargs)

        return cls.cell


@dataclass
class Block:
    block: list[tuple[Cell, int]]


@dataclass
class Layer:
    layer: list[tuple[Block, int]]


@dataclass
class Unit:
    unit: Cell | Block | Layer
