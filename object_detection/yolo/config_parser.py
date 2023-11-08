"""Configuration parser to parse configuration dictionary and define model."""

from dataclasses import dataclass

from object_detection.yolo.config import (
    BlockConfiguration,
    CellConfiguration,
    Configuration,
    LayerConfiguration,
)
from object_detection.yolo.units import Block, Cell, Layer, Model, Unit


@dataclass
class ConfigurationParser:
    configuration: Configuration

    def get_cell(self, cell_configuration: CellConfiguration) -> Cell:
        return Cell(cell_configuration=cell_configuration)

    def get_block_cells(self, block_configuration: BlockConfiguration) -> Block:
        block = Block(cells=[])

        for cell_config, repeat in block_configuration.block:
            block.add_cells(
                cell=self.get_cell(cell_configuration=cell_config), repeat=repeat
            )

        return block

    def get_layer_cells(self, layer_configuration: LayerConfiguration) -> Layer:
        for block_config, repeat in layer_configuration.layer:
            layer = Layer(cells=[])
            block = self.get_block_cells(block_configuration=block_config)
            layer.add_cells(block=block, repeat=repeat)

    def get_model(self) -> Model:
        model = Model(backbone=[])

        for unit_config in self.configuration.unit_configs:
            if isinstance(unit_config, CellConfiguration):
                cell = self.get_cell(cell_configuration=unit_config)
                model.add_unit(unit=Unit(unit=cell))
            elif isinstance(unit_config, BlockConfiguration):
                block = self.get_block_cells(block_configuration=unit_config)
                model.add_unit(unit=Unit(unit=block))
            elif isinstance(unit_config, LayerConfiguration):
                layer = self.get_layer_cells(layer_configuration=unit_config)
                model.add_unit(unit=Unit(unit=layer))

        return model
