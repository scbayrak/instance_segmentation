import os
import torch
import torchvision
from torchvision import transforms
from selim_dataset import COCO_data
from selim_dataset import transform
import torch.optim as optim
import time
from torch.utils import data as data
from PIL import Image as PILImage
import numpy as np
import skimage.io as IO
from selim_utils import get_predictions, apply_segmentation
import cv2

# to resolve error
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

selected_classes = ['person', 'bicycle', 'parking meter']


coco_data = COCO_data("train", selected_classes)
train_loader = data.DataLoader(coco_data, batch_size=1, shuffle=True)
# model_args = {'num_classes':81, 'min_size':1280, 'max_size':1280}
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)

# Training

model.train()
optimizer = optim.Adam(list(model.parameters()), lr=1e-5)

total_epochs = 1

start_time = time.time()

for epoch in range(total_epochs):
    epoch_loss = 0
    no_images = 0
    for X,y in train_loader:
        optimizer.zero_grad()
        if device==torch.device('cuda'):
            X = X.to(device)
            for key in y:
                y[key] = y[key].to(device)
        images = [image for image in X]
        no_images += len(images)
        labels = []
        for key in y:
            y[key] = y[key].squeeze_(0)
        labels.append(y)
        if len(labels) > 0:
            loss = model(images, labels)
            total_loss = sum(loss.values())
            epoch_loss += total_loss.detach().clone().cpu().numpy()
            total_loss.backward()
            optimizer.step()
    epoch_loss /= no_images
    print("Loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss))

end_time = time.time()

print("Training took {0:.1f}".format(end_time-start_time))

# Inference

def deploy_model(image_path, threshold):
    model.eval()
    original_image = IO.imread(image_path)
    image = transform(original_image)
    y_pred = model([image])
    masks, boxes, labels = get_predictions(coco_data, y_pred, threshold)
    image_segmented = apply_segmentation(original_image, coco_data, masks, boxes, labels)
    

deploy_model("picture.jpg", 0.5)


# image = image.unsqueeze_(0)



