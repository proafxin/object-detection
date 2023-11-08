"""List of dictionaries with information of layer structure."""


from object_detection.yolo.units import (
    CellType,
    ConvolutionParams,
    MaxPool2dParams,
    UnitType,
)

UNIT1 = {
    "unit_type": UnitType.BLOCK,
    "config": [
        {
            "cell_type": CellType.Convolutional,
            "params": {
                ConvolutionParams.IN_CHANNELS.value: 3,
                ConvolutionParams.KERNEL_SIZE.value: 7,
                ConvolutionParams.OUT_CHANNELS.value: 64,
                ConvolutionParams.STRIDE.value: 2,
                ConvolutionParams.PADDING.value: 3,
            },
        },
        {
            "cell_type": CellType.MaxPool2d,
            "params": {
                MaxPool2dParams.KERNEL_SIZE.value: 2,
                MaxPool2dParams.PADDING.value: 0,
                MaxPool2dParams.STRIDE.value: 2,
            },
        },
    ],
}

UNITS: list[dict[str, UnitType | list | dict]] = [UNIT1]
