import torch
from torchvision.models.vgg import VGG19_Weights
import torchvision.transforms as T 
from torchvision.models import resnet18, ResNet18_Weights,resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
import random

'''
get_transform(): returns the concerned transforms of a choosen CNN backbone with additional capability to add some predefined data augmentation to the data based on the "aug_prob" parameter, set to 40% by default 

'''

def get_transform(model="vgg19",aug_prob = 0.4):
    model_transforms = {
        'r18':ResNet18_Weights.DEFAULT,
        'r50':ResNet50_Weights.DEFAULT,
        'r101':ResNet101_Weights.DEFAULT,
        'vgg19':VGG19_Weights.DEFAULT
    }
    compulsory = model_transforms[model].transforms()
    data_augmentation = T.Compose([
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(p=0.01),
        T.RandomPerspective(distortion_scale=0.4,p=0.5),
        T.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
        T.RandomAffine(degrees=10,translate=(0.05,0.05),scale=(0.95,1.05)),
        T.GaussianBlur(kernel_size=3,sigma=(0.1,1.0))
    ])

    def transform_fn(img):
        if random.random() < aug_prob:
            img = data_augmentation(img)
        img = compulsory(img)
        return img

    return transform_fn


