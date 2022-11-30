import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from   torch import nn
from   torch import optim
from   torch.autograd import Variable
from   torchvision import datasets, transforms, models
from   torchvision.datasets import ImageFolder
import torch.nn.functional as F

from   PIL import Image

from   collections import OrderedDict

import copy
import os
import json
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', default='my_model_checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default=1)
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

load_checkpoint('my_model_checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    im_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_open = Image.open(image)
    image_tf = im_transforms(image_open)
    
    return image_tf


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def main():
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    category_names = load_cat_names(args.category_names)
    img_path = args.filepath
    topk = args.top_k
    
    probabilities = predict(img_path, model, topk)
    a = np.array(probabilities[0][0])
    b = [category_names[str(index + 1)] for index in np.array(probabilities[1][0])]
    categories = []
    #for cat in a:
    #    cat_names = 
    print('Name of the floweer: ', b[0])
    print('Prediction probability: ', a[0])
    
    
if __name__ == "__main__":
    main()
    


