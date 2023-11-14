"""6th unit of YOLOv1"""


from object_detection.yolo.config import (
    BlockConfiguration,
    Configuration,
    DropOutConfiguration,
    LeakyReLUConfiguration,
    LinearConfiguration,
)
from object_detection.yolo.v1.hyperparams import hyperparams

linear_cell1 = LinearConfiguration(
    in_features=hyperparams.final_conv_output_channel
    * hyperparams.final_conv_output_size
    * hyperparams.final_conv_output_size,
    out_features=hyperparams.linear_output_features,
)
linear_cell2 = LinearConfiguration(
    in_features=hyperparams.linear_output_features,
    out_features=hyperparams.grid_size
    * hyperparams.grid_size
    * (hyperparams.num_class + hyperparams.num_boxes * 5),
)

block = BlockConfiguration(block=[])
block.add_single_cells(
    linear_cell1,
    DropOutConfiguration(p=0),
    LeakyReLUConfiguration(negative_slope=0.1),
    linear_cell2,
)

configuration = Configuration(unit_configs=[])
configuration.add_unit(unit_config=block)
