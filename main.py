from torch.utils.data import DataLoader
import torch

from website_screenshot_dataset import WebsiteScreenshotsDataset
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.utils import draw_bounding_boxes

from vision.engine import train_one_epoch
from PIL import Image
import vision.utils
import cv2
import numpy as np


# The following code for training is derived and adapted from the pytorch tutorial
# on training an RCNN
#
# - https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together
#
# Code changes applied for this project:
# - Epoch number is increased to 50
# - Both datasets use a batch size of 1 instead of 2
# - Instead of using Mask R-CNN this project uses faster-rcnn 
#   pretrained model with a RESNET-50 backbone
# - Our training dataset isn't a variation of our testing data set, 
#   it is instead a completely different set of inputs
def train(device: torch.device):
  transforms = [T.PILToTensor(), T.ConvertImageDtype(torch.float)]
  transfer = T.Compose(transforms)

  dataset = WebsiteScreenshotsDataset(root_dir="./data/train", annotations_filename="_annotations.coco.json", transforms=transfer)

  indices = torch.randperm(len(dataset)).tolist()
  dataset = Subset(dataset, indices[:-50])

  # Load in data from the website screenshot dataset
  data_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=vision.utils.collate_fn)

  # For this we are using the Faster RCNN with the resnet-50 backbone
  model = fasterrcnn_resnet50_fpn()

  # move model to the right device
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
  # and a learning rate scheduler
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

  num_epochs = 20

  for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()

  # Save the model so it can be reused
  torch.save(model.state_dict(), "model/screenshot_model.pt")


def resolve_class_name(class_num: int):
  if class_num == 0:
    return "Element"
  elif class_num == 1:
    return "Button"
  elif class_num == 2:
    return "Field"
  elif class_num == 3:
    return "Heading"
  elif class_num == 4:
    return "iFrame"
  elif class_num == 5:
    return "Image"
  elif class_num == 6:
    return "Label"
  elif class_num == 7:
    return "Link"
  elif class_num == 8:
    return "Text"

def load_model(model_path: str, device: torch.device) -> FasterRCNN:
  # Our model was trained on the GPU, so we to check if it's 
  # being loaded on the CPU
  model = fasterrcnn_resnet50_fpn()
  model.load_state_dict(torch.load(model_path, map_location=device.type))
  model.to(device)
  model.eval()
  return model

def predict_elements(imgs: list[str], model: FasterRCNN):
  converted_imgs = []  

  for img in imgs:
    cur_img = Image.open(img).convert("RGB")
    cur_img = T.PILToTensor()(cur_img)
    cur_img = cur_img / 255
    converted_imgs.append(cur_img)

  return model(converted_imgs)

def show_annotated_imgs(imgs: list[str], boxes: list[torch.Tensor], labels: list[torch.Tensor]):
  for i, img in enumerate(imgs):
    cur_img = Image.open(img).convert("RGB")
    cur_img = T.PILToTensor()(cur_img)
    labeled_labels = list(map(lambda x: resolve_class_name(x), labels[i]))
    annotated_img = T.ToPILImage()(draw_bounding_boxes(cur_img, boxes[i], labels=labeled_labels))
    cv2.imshow('', np.array(annotated_img))
    cv2.waitKey(0)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Uncomment to train model
    train(device)

    model = load_model("./model/screenshot_model.pt", device)

    test_imgs = [
      "./data/test/podcasts_apple_com_png.rf.WOu7SqEveIQD43tzjnKH.jpg",
      "./data/test/npmjs_com_png.rf.N70jnPn3WRmI22F2a7kH.jpg",
      "./data/test/cargocollective_com_png.rf.SgTyNdyw8ipTbCVQjmiy.jpg"
    ]

    annotations = predict_elements(test_imgs, model)

    boxes = list(map(lambda x: x['boxes'], annotations))
    labels = list(map(lambda x: x['labels'], annotations))

    show_annotated_imgs(test_imgs, boxes, labels)