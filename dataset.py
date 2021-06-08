import time
import os, sys, re
import torch
import skimage.io as IO
import matplotlib.pyplot as plt

from torchvision import transforms as Transforms
from torch.utils import data as Data
from pycocotools.coco import COCO

class CocoDataset(Data.Dataset):
    """
    Dataset interface to Object Detection and Segmentation Model
    """
    def __init__(self, stage , categories=['pizza'], coco_path = ""):
        """
        constructor of the dataset class
        Args:
            stage ([string]):  train | test. 
            categories (list, optional): coco dataset categories to be loaded for training. Defaults to ['pizza'].
            coco_path (str, optional): If the annotations json file is in different folder other than dataset/. Defaults to "".
        """       

        if stage == 'train':
            self.coco = COCO(coco_path + "dataset/annotations/instances_train2017.json")
        else:
            self.coco = COCO(coco_path + "dataset/annotations/instances_val2017.json")

        coco_all_cats_idx = self.coco.getCatIds()
        coco_categories_names = self.coco.loadCats(coco_all_cats_idx)

        self.selected_cats_idx = self.coco.getCatIds(catNms=categories)
        coco_images_idx = self.coco.getImgIds(catIds=self.selected_cats_idx)

        self.dataset = self.coco.loadImgs(coco_images_idx)

    @staticmethod
    def transform(image):
        """
        Image processing method.
        Args:
            image ([image]): image to be processed
        Returns:
            [tensor]:  Processed image as tensor
        """        
        # normalization
        t_ = Transforms.Compose([ Transforms.ToPILImage(),
                                  Transforms.ToTensor(),
                                #   Transforms.Normalize(
                                #   mean=[0.485, 0.457, 0.407],
                                #   std=[1,1,1])   
                                ])
        return t_(image)

    def load_image(self, idx):
        """
        This method will bring image as skimage 
        Args:
            idx ([integer]): coco id of the selected image.
        """
        image = IO.imread(self.dataset[idx]['coco_url'])
        if len(image.shape) == 3:
            image = self.transform(image)
        else:
            image = "grayscale"
        return image

    def load_image_labels(self, idx):
        """ This method gets the labels of the given image
            It uses COCO api, get annotation of the image , then
            it extracts categories , boxes and maskes and generate a dict. 
            we use this dictionary , because training model requires targets in this format.

        Args:
            idx ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_id = self.dataset[idx]['id']
        annotation_ids = self.coco.getAnnIds(imgIds=image_id, 
                         catIds=self.selected_cats_idx, iscrowd=None)
        image_annotations = self.coco.loadAnns(annotation_ids)
        image_categories = []
        image_boxes = []
        image_masks = []

        for annotation in image_annotations:
            image_categories.append(annotation['category_id'])
            bbox = annotation['bbox']
            # x, y, width, height => ( coordinates ) x, y, x + width , y + height
            image_boxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            image_masks.append(self.coco.annToMask(annotation))

        image_categories = torch.as_tensor(image_categories, dtype=torch.int64)
        image_boxes = torch.as_tensor(image_boxes, dtype=torch.float32)
        image_masks = torch.as_tensor(image_masks, dtype=torch.uint8)
        image_labels = {}
        image_labels['labels'] = image_categories
        image_labels['boxes'] = image_boxes
        image_labels['masks'] = image_masks
        return image_labels

    def __len__(self):
        """
        Magic method that is called in Dataloader iterations. 
        Returns:
            [type]: [description]
        """        
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Magic method that is called in Dataloader iterations. 
        Args:
            idx ([integer]): index of the image in the dataset
        Returns:
            [image, labels]: tensor, labels are python dict, keys are labels, boxes, masks
        """        
        image =  self.load_image(idx)
        labels = self.load_image_labels(idx)
        return image, labels
