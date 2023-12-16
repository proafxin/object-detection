"""Some parameters for reading data."""


from enum import Enum


class AnnotationFormat(str, Enum):
    YOLO = "yolo"
    PASCALVOC = "pascalvoc"
    COCO = "coco"


IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg"]
ANNOTATION_FORMATS: dict[str, str] = {
    ".txt": AnnotationFormat.YOLO.value,
    ".xml": AnnotationFormat.PASCALVOC.value,
    ".json": AnnotationFormat.COCO.value,
}
