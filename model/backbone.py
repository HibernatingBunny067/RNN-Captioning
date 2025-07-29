import torch
import torch.nn as nn
# from data.dataset import CaptionDataset
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    vgg19, VGG19_Weights
)
from torchinfo import summary

'''
CNNBackbone(): primary class required to obtain the CNN encoder from some predefined choices used and experimented with in this project 
'''



class CNNBackbone(nn.Module):
    WEIGHTS = {
        'vgg19':VGG19_Weights.DEFAULT,
        'r50':ResNet50_Weights.DEFAULT,
        'r18':ResNet18_Weights.DEFAULT,
        'r101':ResNet101_Weights.DEFAULT
    }
    MODELS = {
      'vgg19':vgg19,
        'r50':resnet50,
        'r18':resnet18,
        'r101':resnet101
    }
    def __init__(self,model ='r50',attention=False,remove_fc=True,freeze=True,device='cpu'):
        super().__init__()
        self.device = device
        self.model = model
        self.attention = attention
        self.weights = CNNBackbone.WEIGHTS[self.model]
        self.initialized_model = CNNBackbone.MODELS[self.model](weights = self.weights).to(self.device)
        if remove_fc:
            self.remove_classifier_head()
        if freeze:
            self.freeze_backbone()
        if self.attention:
            self.isAttentive()

    def remove_classifier_head(self):
        self.initialized_model.fc = nn.Identity()

    def isAttentive(self):
      modules = list(self.initialized_model.children())
      if self.model != 'vgg19':
        take = modules[:-2]
      else:
        take = modules[0][:36] 
      self.initialized_model = nn.Sequential(*take)

    def get_model(self):
        return self.initialized_model
    
    def freeze_backbone(self):
        for params in self.initialized_model.parameters():
            params.requires_grad = False

    def get_feature_dim(self,input_size=(3,224,224),batch_size=1):
        dummy = torch.randn(batch_size,*input_size,device=self.device)
        out = self.forward(dummy)
        return out.shape[-1] if not self.attention else out.shape[1:]
            

    def print_summary(self,batch_size=100):
        return summary(self.initialized_model,(1,3,224,224))
    
    def forward(self,X):
        with torch.no_grad():
            out = self.initialized_model(X)
            if self.attention:
                B,C,H,W = out.shape
                out = out.view(B,C,H*W).permute(0,2,1)
            else:
                out = out.view(out.size(0),-1)
            return out 


            
        


