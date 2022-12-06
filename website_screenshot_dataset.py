from coco_types import COCOJSON
from torch.utils.data import Dataset
from os.path import join
import json
import torch
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms as T

# Loading COCO format image annotations in pytorch is somewhat tedious.
# As a result I consulted online resources with code examples on how to
# translate this format into something that PyTorch's models can properly
# process.
# 
# 
# This file is adapted from the following code sources.
#
# - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html By Susank Chilamkurty
# - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html By PyTorch
# - https://medium.com/fullstackai/how-to-train-an-object-detector-with-your-own-coco-dataset-in-pytorch-319e7090da5 By Dr. Takashi Namura

class WebsiteScreenshotsDataset(Dataset):

  root_dir: str
  annotations: COCOJSON
  coco: COCO

  def __init__(self, root_dir: str, annotations_filename: str, transforms) -> None:
    # Store a reference to the root directory of the data
    self.root_dir = root_dir

    # Load in the annotations for each of the images as a dictionary.
    annotations_file = open(join(root_dir, annotations_filename))
    self.annotations = json.load(annotations_file)
    self.coco = COCO(annotation_file=join(root_dir, annotations_filename))
    self.transforms = transforms

  def get_annotations_for_image(self, image_id: int):
    return [annotation for annotation in self.annotations['annotations'] if annotation['image_id'] == image_id]

  def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    current_image = self.annotations['images'][index]
    annotations = self.get_annotations_for_image(current_image['id'])
    categories = self.annotations['categories']

    image_filepath = join(self.root_dir, self.annotations['images'][index]['file_name'])
    image = Image.open(image_filepath).convert("RGB")

    boxes = []
    labels = []
    for annotation in annotations:
      origin_x, origin_y, box_width, box_height = annotation['bbox']
      
      # We need to recalculate the values since pytorch models don't use this format
      min_x = origin_x
      min_y = origin_y
      max_x = min_x + box_width
      max_y = min_y + box_height

      if (min_x >= max_x or min_y >= max_y):
        continue

      found_category = list(filter(lambda x: x['id'] == annotation['category_id'] ,categories))[0]
      labels.append(categories.index(found_category))
      boxes.append([min_x, min_y, max_x, max_y])
    
    # Convert to tensor
    area = torch.as_tensor(list(map(lambda x: x['area'], annotations)))
    is_crowd = torch.zeros(len(annotations), dtype=torch.float32)

    target = {
      "boxes": torch.as_tensor(boxes),
      "image_id": torch.as_tensor([current_image['id']]),
      "area": area,
      "iscrowd": is_crowd,
      "labels": torch.as_tensor(labels)
    }

    return T.ToTensor()(image), target

  def __len__(self):
    return len(self.annotations['images'])