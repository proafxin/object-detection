"""Read images and annotations"""

from dataclasses import dataclass, field

from torchvision.datasets import ImageFolder


@dataclass
class ObjectDetectionDataset:
    image_dataset: ImageFolder = field(init=False)
