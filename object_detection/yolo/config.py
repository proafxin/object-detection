"""List of dictionaries with information of layer structure."""


from pydantic import BaseModel, Field


class UnitConfiguration(BaseModel):
    """Base cell configuration class.
    All base cell configuration class should inherit this.
    This class specifies all necessary fields for the cell.
    Whenever possible, specify a default value for a field.
    """


class CellConfiguration(UnitConfiguration):
    pass


class DropOutConfiguration(CellConfiguration):
    p: float = Field(default=0.2)
    inplace: bool = Field(default=False)


class LinearConfiguration(CellConfiguration):
    in_features: int
    out_features: int
    bias: bool = Field(default=False)


class FlattenConfiguration(CellConfiguration):
    start_dim: int = Field(default=0)
    end_dim: int = Field(default=-1)


class BatchNorm2dConfiguration(CellConfiguration):
    num_features: int
    eps: float = Field(default=1e-05)
    momentum: float = Field(default=0.1)


class MaxPool2dConfiguration(CellConfiguration):
    kernel_size: int = Field(default=2)
    stride: int = Field(default=2)
    padding: int = Field(default=0)


class Convolution2dConfiguration(MaxPool2dConfiguration):
    in_channels: int
    out_channels: int
    bias: bool = Field(default=False)


class ReLUConfiguration(CellConfiguration):
    inplace: bool = Field(default=False)


class LeakyReLUConfiguration(ReLUConfiguration):
    negative_slope: float = Field(default=0.01)


class BlockConfiguration(UnitConfiguration):
    """Configuration class for `Block`
    Refer to `Block` for more information.
    """

    block: list[tuple[CellConfiguration, int]]

    def add_cells(self, cell_config: CellConfiguration, repeat: int = 1) -> None:
        self.block.append((cell_config, repeat))

    def add_single_cells(self, *cells) -> None:
        for cell in cells:
            self.block.append((cell, 1))

    def extend(self, cell_configs: list[tuple[CellConfiguration, int]]) -> None:
        for cell_config, repeat in cell_configs:
            self.block.append((cell_config, repeat))


class LayerConfiguration(UnitConfiguration):
    layer: list[tuple[BlockConfiguration, int]]

    def add_blocks(self, block_config: BlockConfiguration, repeat: int = 1) -> None:
        self.layer.append((block_config, repeat))

    def add_single_cell(self, cell_config: CellConfiguration, repeat: int = 1) -> None:
        block = BlockConfiguration(block=[])
        block.add_cells(cell_config=cell_config, repeat=repeat)
        self.layer.append((block, repeat))


class Configuration(BaseModel):
    unit_configs: list[UnitConfiguration]

    def add_unit(self, unit_config: UnitConfiguration) -> None:
        self.unit_configs.append(unit_config)
