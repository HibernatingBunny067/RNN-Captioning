import torch
from torch.nn.modules import transformer
from torch.utils.data import DataLoader
from data.dataset import ImageOnlyDataset, FeatureCaptionDataset
from data.collate import Padding
import os
from tqdm.notebook import tqdm
import sys
import numpy as np

'''
GetData( ): Primary class to do the batch pre computation of the feature maps and fetching the data loaders from the stored data during training 
'''

class GetData():
  def __init__(self,save_folder,extractor=None,transform=None,imag_folder=None,file_list=None):
    self.imag_folder = imag_folder
    self.save_folder = save_folder
    self.file_list = file_list
    self.extractor = extractor
    self.transform = transform
    

  def batched_pre_compute(self,batch_size=64,workers=2):
    os.makedirs(self.save_folder,exist_ok =True)
    dataset = ImageOnlyDataset(image_folder=self.imag_folder,file_list=self.file_list,transform=self.transform)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers)
    device = self.extractor.device
    count= 0
    for image,img_id in tqdm(loader):
      images = image.to(device)
      with torch.no_grad():
        features = self.extractor.forward(images)
      features = features.detach().to('cpu')
      #remove the computational graph, gradienst, bring to cpu and turn to float32 
      for i in range(features.shape[0]):
        filename = f'{img_id[i].item()}.npy'
        save_path = os.path.join(self.save_folder,filename)
        np.save(save_path,features[i].numpy())
        count+=1
    print(f'Saved {count} features to {self.save_folder}')


  def get_loaders(self,vocab,samples,batch_size=100,shuffle=False,workers =0,persists=True):
    dataset = FeatureCaptionDataset(samples,self.save_folder)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=Padding(vocab.word2idx['<pad>']),
    num_workers=workers,persistent_workers=persists)
    return loader 

