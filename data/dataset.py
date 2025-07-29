import torch
from torch.utils.data import Dataset
from PIL import Image
import os 
import random 
import numpy as np

'''
ImageOnlyDataset(): used to loade image_paths with img_id during batch_precomputation phase

FeatureCaptionDataset(): return feature_map, caption (selected at random from all the possible captions corresponding to a particular image filtered from the 5 MS COCO captions) and imag_ids used to retrieve all the captions during validation 
'''


class ImageOnlyDataset(Dataset): #returns images, id for precomputing feature maps on train and val sets 
  def __init__(self,image_folder,file_list,transform):
    self.image_folder = image_folder
    self.file_list = file_list
    self.transform = transform
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,idx):
    filename = self.file_list[idx][0]
    path = os.path.join(self.image_folder,filename)
    image = Image.open(path).convert('RGB')
    image = self.transform(image)
    img_id = self.file_list[idx][1]
    return image,img_id

class FeatureCaptionDataset(Dataset):
    def __init__(self, samples, features_folder):
        self.samples = samples
        self.features_folder = features_folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['id']
        captions = sample['captions']
        
        caption = random.choice(captions)
        caption = torch.tensor(caption, dtype=torch.long)

        feature_path = os.path.join(self.features_folder, f"{img_id}.npy")
        feature = torch.from_numpy(np.load(feature_path))

        return feature, caption,img_id