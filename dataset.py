from torch.utils import data
from pycocotools.coco import COCO
from torchvision import transforms as transforms
import skimage.io as io
import numpy as np
import torch

def transform(img):
         t_ = transforms.Compose([
                             transforms.ToPILImage(),
                             transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                                  std=[1,1,1])
                             ]) 
  
         img = t_(img)
         return img


class COCO_data(data.Dataset):
    def __init__(self, stage, selected_classes):
        if stage == "train":
            self.interface = COCO("dataset/annotations/instances_train2017.json")
        else:
            self.interface = COCO("dataset/annotations/instances_val2017.json")

        self.selected_category_ids = self.interface.getCatIds(catNms=selected_classes)
        self.adjusted_category_ids = {}
        self.image_nos = self.interface.getImgIds(catIds=self.selected_category_ids)
        self.images = self.interface.loadImgs(self.image_nos)
        self.all_categ_ids = self.interface.getCatIds()
        self.all_categ_names = self.interface.loadCats(self.all_categ_ids)
        self.all_categ_ids.insert(0,0)
        background_category = {'id': 0, 'name': 'background', 'supercategory': 'N/A'}
        self.all_categ_names.insert(0,background_category)

        for new_id, old_id in enumerate(self.all_categ_ids):
            self.adjusted_category_ids[old_id] = new_id
    
    def load_X(self, idx):
        image = np.array(io.imread(self.images[idx]["coco_url"]))
        image = transform(image)
        return image

    def load_y(self, idx):
        image_ids = self.images[idx]["id"]
        annotation_ids = self.interface.getAnnIds(imgIds=image_ids, catIds=self.selected_class_ids, iscrowd=None)
        annotations = self.interface.loadAnns(annotation_ids)
        boxes = []
        category_ids = []
        masks = []
        for an in annotations:
            new_id =  self.adjusted_category_ids[an["category_id"]]
            category_ids.append(new_id)
            # category_ids.append(an["category_id"])
            box = an["bbox"]
            #"bbox": [x,y,width,height],
            box = [box[0], box[1], box[0]+box[2], box[1] + box[3]]
            boxes.append(box)
            mask = self.interface.annToMask(an)
            masks.append(mask)
        boxes = torch.as_tensor(boxes, dtype = torch.float)
        ids = torch.tensor(ids, dtype=torch.int64)
        masks = torch.tensor(masks, dtype=torch.uint8)
        labels = {}
        labels["boxes"] = boxes
        labels['labels'] = category_ids
        labels['masks'] = masks
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.load_X(idx)
        y = self.load_y(idx)
        return X, y

