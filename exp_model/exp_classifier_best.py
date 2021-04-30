import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.models as models
import torchvision.utils
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    
import os

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    print(npimg.shape)
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


class EmotionRecognition(nn.Module):
    def __init__(self):
        super(EmotionRecognition, self).__init__()
        self.conv1=nn.Conv2d(896,512, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)
      
        self.conv2=nn.Conv2d(512, 256,kernel_size=1, stride=1)
        self.mp1=nn.MaxPool2d(7,1)
        self.conv3=nn.Conv2d(256,256,kernel_size=1,stride=1)
        self.conv4=nn.Conv2d(512, 64,kernel_size=1, stride=1)
        self.mp2=nn.MaxPool2d(2,1)
        self.fc1 = nn.Sequential(
            nn.Linear(2504, 1000),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),

            nn.Linear(1000, 200),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(200,4))
        self.bn1 = nn.BatchNorm2d(512, eps=0.001)
        self.bn2= nn.BatchNorm2d(256,eps=1e-05)
    def forward(self, out1,out2,out3,out4):
      
        outa=self.conv1(out2)
        outa=self.bn1(outa)
        outa=self.relu(outa)
        
        
        outa=self.conv2(outa)
        outa=self.bn2(outa)
        outa=self.relu(outa)
        

        outb=self.conv3(out1)
        outb=self.relu(outb)
        outb=self.mp1(outb)

        outc=torch.cat((outa,outb),1)

        outc=self.conv4(outc)
        outc=self.relu(outc)
        outc=self.mp2(outc)
        outc=torch.flatten(outc,start_dim=1)
        out3=torch.flatten(out3,start_dim=1)
        
        outd=torch.cat((outc,out3,out4),1)
     
        output=self.fc1(outd)
       
        return output