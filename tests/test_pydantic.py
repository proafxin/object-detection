from pydantic import BaseModel, Field
from torch import Tensor
from torch.nn import Conv2d


class Model(BaseModel):
    kernel_size: int = Field(default=2)
    bias: bool = Field(default=False)
    in_channels: int = Field(default=2)
    out_channels: int = Field(default=5)
    stride: int = Field(default=2)
    padding: int = Field(default=0)


def test_pydantic_model(x_cnn: Tensor):
    model = Model(in_channels=3, out_channels=10)
    cnn = Conv2d(**model.model_dump())
    assert isinstance(cnn, Conv2d)
    out = cnn(x_cnn)
    assert isinstance(out, Tensor)
