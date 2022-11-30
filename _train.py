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
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store', default = 'flowers/')
    #parser.add_argument('--arch', dest='arch', default='densenet121')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001)
    #parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default=2)
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def save_checkpoint(path, model, optimizer, args, classifier):
    
    checkpoint = {
                  'arch': args.arch, 
                  'model': model,   
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)
    
def main():
    args = parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    other_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),                                                                                                            transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]) 


    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=other_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = other_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size =32,shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    
    model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    train_data_size = len(train_data)
    valid_data_size = len(valid_data)
    test_data_size = len(test_data)
    
    print('sizes of training, validation and testing datasets are: ', train_data_size, valid_data_size, test_data_size, '\n')
    
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    device = torch.cuda.is_available()
    device, torch.__version__

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...\n')
    else:
        print('CUDA is available!  Training on GPU ...\n')

    model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
        
    lr = args.learning_rate

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([ 
                              ('fc1', nn.Linear(1024, 500)),
                              ('dropout', nn.Dropout(p=0.6)),
                              ('relu1', nn.ReLU()),
                              ('fc2', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    print('model loaded !\n')
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    model.to('cuda')

    epochs     = args.epochs
    print_freq = 20
    steps      = 0
    
    running_start_time = time.time()
    print('model training started...\n')

    for e in range(epochs):
        current_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader): 
            steps = steps + 1 

            inputs, labels = inputs.to('cuda'), labels.to('cuda') 

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

            if steps % print_freq == 0:
                model.eval()
                valid_loss = 0
                accuracy   = 0

                for ii, (inputs2,labels2) in enumerate(valid_loader): 
                        optimizer.zero_grad()

                        inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') 
                        model.to('cuda:0') 
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valid_loss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                valid_loss = valid_loss / len(valid_loader)
                accuracy = accuracy /len(valid_loader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(current_loss/print_freq),
                      "Validation Loss {:.4f}".format(valid_loss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                current_loss = 0
                
    time_elapsed = time.time() - running_start_time
    print('model training finished !\n')
    print("\ntraining time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))   
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        #'arch': args.arch, 
        'model': model,   
        'learning_rate': args.learning_rate,
        #'hidden_units': args.hidden_units,
        'classifier' : classifier,
        'epochs': args.epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'my_model_checkpoint.pth')
            
if __name__ == "__main__":
    main()