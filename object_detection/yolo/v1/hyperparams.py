"""Some hyperparameters"""


from pydantic import BaseModel, Field


class HyperParameters(BaseModel):
    image_size: int = Field(default=448)
    grid_size: int = Field(default=19)
    final_conv_output_channel: int = Field(default=1024)
    final_conv_output_size: int = Field(default=3)
    num_class: int = Field(default=20)
    linear_output_features: int = Field(default=4096)
    num_boxes: int = Field(default=2)
    in_channels: int = Field(default=3)


hyperparams = HyperParameters()
