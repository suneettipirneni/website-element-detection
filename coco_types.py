from typing import TypedDict


class COCOLicenseData(TypedDict):
  id: int
  name: str
  url: str

class COCOImageData(TypedDict):
  id: int
  width: int
  height: int
  file_name: str
  license: int
  date_captured: str

class COCOInfoData(TypedDict):
  year: int
  version: float
  description: str
  contributor: str
  url: str
  date_created: str

class COCOAnnotation(TypedDict):
  id: int
  image_id: int
  category_id: int
  bbox: tuple[int, int, int, int]
  area: int
  segmentation: list
  iscrowd: int

class COCOCategoryData(TypedDict):
  id: int
  name: str
  supercategory: str
  isthing: int
  color: tuple[int, int, int]

class COCOJSON(TypedDict):
  info: COCOImageData
  licenses: list[COCOLicenseData]
  images: list[COCOImageData]
  categories: list[COCOCategoryData]
  annotations: list[COCOAnnotation]