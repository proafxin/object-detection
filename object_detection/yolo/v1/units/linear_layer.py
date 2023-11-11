"""6th unit of YOLOv1"""


from object_detection.yolo.config import (
    BlockConfiguration,
    Configuration,
    DropOutConfiguration,
    LeakyReLUConfiguration,
    LinearConfiguration,
)
from object_detection.yolo.v1.settings import (
    FINAL_CONV_OUTPUT_CHANNEL,
    FINAL_CONV_OUTPUT_SIZE,
    GRID_SIZE,
    LINEAR_OUTPUT_FEATURES,
    NUM_BOXES,
    NUM_CLASS,
)

linear_cell1 = LinearConfiguration(
    in_features=FINAL_CONV_OUTPUT_CHANNEL
    * FINAL_CONV_OUTPUT_SIZE
    * FINAL_CONV_OUTPUT_SIZE,
    out_features=LINEAR_OUTPUT_FEATURES,
)
linear_cell2 = LinearConfiguration(
    in_features=LINEAR_OUTPUT_FEATURES,
    out_features=GRID_SIZE * GRID_SIZE * (NUM_CLASS + NUM_BOXES * 5),
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
