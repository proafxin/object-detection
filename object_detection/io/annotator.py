"""Annotation related functions."""


from dataclasses import dataclass
from os.path import splitext

from pydantic import BaseModel, Field

from object_detection.io.settings import ANNOTATION_FORMATS, AnnotationFormat


class BoundingBox(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    label: str


class Source(BaseModel):
    database: str | None = Field(default=None)
    annotation: str | None = Field(default=None)
    image: str | None = Field(default=None)


class Annotation(BaseModel):
    directory: str
    source: Source | None = Field(default=None)
    height: int
    width: int
    depth: int
    segmented: bool = Field(default=False)
    objects: list[BoundingBox]


@dataclass
class AnnotationReader:
    def _get_format(self, path: str) -> str:
        filename, ext = splitext(path)
        if ext in ANNOTATION_FORMATS:
            return ANNOTATION_FORMATS[ext]

        raise KeyError(f"{ext} not in {list(ANNOTATION_FORMATS.keys())}")

    def read_annotation(self, path: str, format: str | None) -> Annotation:
        if not format:
            format = self._get_format(path=path)

        if format not in AnnotationFormat:
            raise KeyError(f"{format} is not a valid format.")
