# Object Detection

Library of different object detection algorithms. You need a CUDA compatible GPU for this to work.

[![Trunk Check](https://github.com/proafxin/object-detection/actions/workflows/trunk.yml/badge.svg)](https://github.com/proafxin/object-detection/actions/workflows/trunk.yml)

[![Upload Code Coverage to Codecov](https://github.com/proafxin/object-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/proafxin/object-detection/actions/workflows/ci.yml)

## Development environment

This project uses trunk for linting and formatting. Check <https://trunk.io/products/check>

For dependency, `poetry` is used: <https://python-poetry.org/>

Go to root directory and run `poetry install`. But first make sure you have CUDA in your system along with all necessary packages. You can follow <https://github.com/proafxin/cuda-gpu-scripts> for CUDA related instructions. Make sure CUDA can interact with your GPU. The best way to ensure that is to go to python shell and run

```python
import torch
torch.cuda.is_available()
```

You should see `True` as output. Finally, for virtual environment management, automated testing and task management, `tox` is used: <https://tox.wiki/>

Install `tox` in your system using `python3 -m pip install -U tox` and then run `tox` from the project directory.
