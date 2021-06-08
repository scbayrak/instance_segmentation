import time
import os, sys, re
import random
import numpy as np
import torch
import torchvision
import torch.optim as Optim
from torch.utils import data
import torchvision.transforms as T
import json

import dataset as Dataset
import utils as Utils
import evaluation

from pycocotools.coco import COCO
from PIL import Image

import matplotlib.pyplot as plt

from cv2 import cv2
from tqdm import tqdm


MODEL_WEIGHTS_FOLDER = "model_weights/"
MODEL_WEIGHTS_EXTENSION = ".pth"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Mask_RCNN_Model():
    """
    Main model implementation for object detection and segmentation
    """
    def __init__(self,model_name , pretrained=True, categories=[], model_save_path=""):
        """
        model constructor

        Args:
            model_name ([type]): This is used to save or load from file of weights and stats. 
            pretrained (bool, optional):  set it True if one desires to use pretraining weights. Defaults to True.
            categories (list, optional): Cocodataset categories. Defaults to [].
            model_save_path (str, optional): It is used to add additional path if weights are different folder such as Google Drive etc.. Defaults to "".
        """
        self.pretrained = pretrained
        self.device = DEVICE
        self.categories = categories
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.stats = {}
        

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=self.pretrained).to(self.device)
        if self.pretrained:
            self.ready_for_inference_flag = True
        else: 
            self.ready_for_inference_flag = False

    def __generate_dataloaders(self, loader="train"):
        """
        Generate test and validation dataloader from dataset interface

        Args:
            loader (str, optional):. Defaults to "train".
        """
        if loader == 'train':
            self.train_dataset = Dataset.CocoDataset(stage='train', categories=self.categories, coco_path=self.model_save_path)
            self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        if loader == 'validation' or loader == 'test':
            self.test_dataset = Dataset.CocoDataset(stage='validation', categories=self.categories, coco_path=self.model_save_path)
            self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=1, shuffle=True)

    def __save_model(self, suffix=""):
        """
        Save model parameters and stats to pth and json file. 

        Args:
            suffix (str, optional): It is used to differenciate each epochs file for checkpointing.. Defaults to "".
        """
        self.file_path = MODEL_WEIGHTS_FOLDER + self.model_name + suffix +  MODEL_WEIGHTS_EXTENSION
        if self.model_save_path != "":
            self.file_path = self.model_save_path + self.file_path
        
        torch.save(self.model.state_dict(), self.file_path)

        self.stats_file_path = MODEL_WEIGHTS_FOLDER + self.model_name + suffix +  '.json'
        if self.model_save_path != "":
            self.stats_file_path = self.model_save_path + self.stats_file_path
        
        with open(self.stats_file_path , 'w') as fp:
            json.dump(self.stats, fp)


    def __load_model(self ):
        """
        Loads weights from file , filename is generated with model name at the time of instantiation of class. 

        Returns:
            [Bool]:  True if it is succesfully loaded. 
        """
        self.file_path = MODEL_WEIGHTS_FOLDER + self.model_name + MODEL_WEIGHTS_EXTENSION
        if self.model_save_path != "":
            self.file_path = self.model_save_path + self.file_path

        if os.path.exists(self.file_path):
            self.model.load_state_dict(torch.load(self.file_path, map_location=DEVICE))
            self.model.eval()
            return True
        else:
            return False
        
    
    def predict(self, img_path='output/pizza.jpg', confidence=0.7):
        """
        Prediction method given input image and confidence 

        Args:
            img_path (str, optional): [description]. Defaults to 'output/pizza.jpg'.
            confidence (float, optional): [description]. Defaults to 0.7.
        """
        if self.ready_for_inference_flag == False:
            print("Model is not pretrained  nor trained on a dataset, please use pretrained model or train the model with dataset before inference")
        else:
            self.model.eval()  
            img = Utils.load_image(img_path)
            img = img.to(self.device)
            start_time = time.time()
            prediction = self.model([img])
            end_time = time.time()
            print(f"Time taken for inference:{end_time - start_time}")
            Utils.segment_instance(prediction, img_path, confidence=confidence, text_thickness=1)

    def train(self, load_existing_weights=False, epochs=5, save_every_x_epoch=5, continue_training=False, epoch_from_loaded_file=0):
        """
        Training method. 

        Args:
            load_existing_weights (bool, optional): It is a flag to use existing weights from checkpoint file. Defaults to False.
            epochs (int, optional): Number of training epochs. Defaults to 5.
            save_every_x_epoch (int, optional): Checkpoint for every x epochs. . Defaults to 5.
            continue_training (bool, optional): If the training is interrupted, this parameter is used to load last weights and continue_training. Defaults to False.
            epoch_from_loaded_file (int, optional): This is for stats to track the epoch loss if the previous training is interrupted. . Defaults to 0.

        Returns:
            [type]: [description]
        """
        # if the model is pretrained, no need training. 
        if self.pretrained:
            print("Model is already pretrained, no need for training!")
            return False

        if load_existing_weights:
            if self.__load_model():
                self.ready_for_inference_flag = True
                print("Custom weights are loaded!")
                if continue_training == False:
                    return True
            else:
                print("No model weights exists, please train the model first, or use pretrained model")
                return False

        # else training will be performed. 
        self.__generate_dataloaders()
        optimizer = Optim.Adam(list(self.model.parameters()), lr=1e-5)
        losses= []
        mAPs = []
        mARs = []
        for epoch in range(epochs):
            self.model.train()
            print("epoch", epoch)
            epoch_loss = 0
            for X,y in tqdm(self.train_dataloader):
                try:
                    self.model.zero_grad()
               
                    X = X.to(self.device)
                    for key in y:
                        y[key] = y[key].to(self.device)

                    images = [img for img in X]
                    targets = []
                    target = {}
                    for key in y:
                        target[key] = y[key].squeeze_(0)
                    targets.append(target)

                    loss = self.model(images, targets)
                    total_loss = sum(loss.values())
                    epoch_loss += total_loss.item()
                    total_loss.backward()        
                    optimizer.step()
                except:
                   print("image skipped", X)

            epoch_loss = epoch_loss/len(self.train_dataloader)
            print("Loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss))
            losses.append(epoch_loss)
            mAP, mAR = self.evaluation()
            mAPs.append(mAP)
            mARs.append(mAR)
            self.stats['Loss'] = losses
            self.stats['mAP'] = mAPs
            self.stats['mAR'] = mARs
            
            self.__save_model(suffix="aftereveryepoch")
            if epoch % save_every_x_epoch == 0:
                self.__save_model(suffix=str(epoch+epoch_from_loaded_file))
                print(self.stats)

        # Testing 
        self.__save_model(suffix="999")
        Utils.evolution_metric_plot(self.stats)

    def evaluation(self):
        """
        Generates evoluation metric values such as mAP and mAR

        Returns:
            [type]: [description]
        """
        # create torch tensors to keep all MAP and MAR values
        self.__generate_dataloaders(loader='test')
        all_MAPs, all_MARs = np.zeros((2, len(self.test_dataloader)))
        self.model.eval()
        for id, data in tqdm(enumerate(self.test_dataloader)):
            X,y = data
            if X[0] != "grayscale":
                if len(y["boxes"]):
                    for key in y:
                        y[key] = y[key].to(self.device)
                    X = X.to(self.device)
                    images = [img for img in X]
                    prediction = self.model(images)
                    # we have test labels and predicted labels. 

                    pred_boxes = prediction[0]['boxes'].detach()
                    pred_class_ids = prediction[0]["labels"].detach()
                    pred_scores = prediction[0]["scores"].detach()
                    pred_masks = prediction[0]["masks"].detach()
                    gt_boxes = y["boxes"].squeeze().detach()
                    gt_class_ids = y["labels"].squeeze().detach()
                    gt_masks = y["masks"].squeeze().detach()

                    if len(pred_masks) != 0:
                        MAP = evaluation.compute_ap_iou_range(
                            gt_boxes, gt_class_ids, gt_masks, pred_boxes, 
                            pred_class_ids, pred_scores, pred_masks)

                        MAR = evaluation.compute_recall_iou_range(
                            pred_masks, gt_masks)

                        all_MAPs[id] = MAP

                        all_MARs[id] = MAR

        Mean_Average_Precision = round(np.mean(all_MAPs), 3)
        Mean_Average_Recall = round(np.mean(all_MARs), 3)
        print("mAP:\t" , Mean_Average_Precision , "mAR:\t", Mean_Average_Recall)
        self.ready_for_inference_flag = True
        return Mean_Average_Precision, Mean_Average_Recall
