"""A single unit of a block."""

from dataclasses import dataclass

from torch.nn import BatchNorm2d, Conv2d, LeakyReLU, MaxPool2d, Module, ReLU, Sequential

from object_detection.yolo.config import (
    BatchNorm2dConfiguration,
    CellConfiguration,
    Convolution2dConfiguration,
    LeakyReLUConfiguration,
    MaxPool2dConfiguration,
    ReLUConfiguration,
)


@dataclass
class Cell:
    """
    Cell is the most basic unit in the model.
    Example: conv2d, maxpool2d, relu, leakyrelu, etc.
    Based on the cell configuration, this constructor returns the appropriate instantiation.
    """

    def __new__(cls, cell_configuration: CellConfiguration) -> Module:
        kwargs = cell_configuration.model_dump()

        if isinstance(cell_configuration, BatchNorm2dConfiguration):
            return BatchNorm2d(**kwargs)

        if isinstance(cell_configuration, Convolution2dConfiguration):
            return Conv2d(**kwargs)

        if isinstance(cell_configuration, MaxPool2dConfiguration):
            return MaxPool2d(**kwargs)

        if isinstance(cell_configuration, LeakyReLUConfiguration):
            return LeakyReLU(**kwargs)

        if isinstance(cell_configuration, ReLUConfiguration):
            return ReLU(**kwargs)


@dataclass
class Block:
    """A single block consists of some cells.
    One cell is sometimes repeated multiple times at one go.
    So the structure is a list of tuples.
    Each tuple has two elements.
    First element has the cell configuration.
    Second element is an integer specifying the number of times this cell should be repeated.
    An example: [(Conv2d, 3), (MaxPool2d, 1)]
    Here, first cell is a convolution unit which should be repeated 3 times.
    Then a maxpool unit which should not be repeated.
    Note: `cells` is a direct list of all the cells.
    Meaning for the example above, it will be like this: [Conv2d, Conv2d, Conv2d, MaxPool2d]
    This is made so that we can access the cells directly without extra computation.
    This way building the sequential model is easier.
    """

    cells: list[Cell]

    def add_cells(self, cell: Cell, repeat: int) -> None:
        """Add new cells to the current block.
        This function helps add more cells in a compressed manner without adding them manually.

        Parameters
        ----------
        cell : ``Cell``
            The cell to be added.
        repeat : ``int``
            The number of times to be added.
        """
        for _ in range(repeat):
            self.cells.append(cell)


@dataclass
class Layer:
    """Usually big neural networks have multiple `layers`.
    Each such layer has multiple blocks.
    Sometimes one block is repeated multiple times.
    We added cells repeatedly in a block to easily add them to the model.
    Here we add blocks repeatedly to add them easily to the model.
    """

    cells: list[Cell]

    def add_cells(self, block: Block, repeat: int = 1) -> None:
        """Adds new cells to the current block.
        This function helps add more cells in a compressed manner without adding them manually.

        Parameters
        ----------
        block : ``Block``
            Block to be added.
        repeat : ``int``
            Number of times the block is repeated.
        """
        for _ in range(repeat):
            self.cells.extend(block.cells)

    def add_single_cell(self, cell: Cell, repeat: int = 1) -> None:
        for _ in range(repeat):
            self.cells.append(cell)


@dataclass
class Unit:
    """We asssume that each model consists of some `units`.
    Such a unit can have a single cell or a layer or a single block.
    """

    unit: Cell | Block | Layer

    def get_cells(self) -> list[Cell]:
        """Makes adding cells to the model easy.

        Returns
        -------
        ``list[Cell]``
            List of cells in current unit.
        """
        if isinstance(self.unit, Cell):
            return [self.unit]

        return self.unit.cells


@dataclass
class Model:
    backbone: list[Unit]

    def add_unit(self, unit: Unit) -> None:
        self.backbone.append(unit)

    def all_cells(self) -> list[Cell]:
        cells = []
        for unit in self.backbone:
            cells.extend(unit.get_cells())

        return cells

    def get_sequential(self) -> Sequential:
        return Sequential(*self.all_cells())
