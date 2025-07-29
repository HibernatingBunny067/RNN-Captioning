import matplotlib.pyplot as plt 
import random 
from PIL import Image
import os

'''
show_data( ): used to visualize the data-sample from the MS COCO 
- temporarily used in training and preprocessing
- can only be used if all the images are saved somewhere in the local disk storage 
'''


def show_data(data_sample,root,vocab,r=2,size=12,encoded=False):
  ## input the data samples and enjoy the visualization of the data
    if r < 2:
        r = 2
    elif type(r) != int:
        r = int(r)
    select = r*r
    sample_selected = random.sample(data_sample,select)
    fig,axis = plt.subplots(r,r,figsize=(size,size))
    axis =axis.flatten()
    for i,dicts in enumerate(sample_selected):
      img = Image.open(os.path.join(root,dicts['path'])).convert('RGB')
      caption = random.choice(dicts['captions'])
      if  encoded:
        cap = [token for token in caption if token!=vocab.word2idx['<pad>']]
        text =vocab.decode(cap)
      else:
        text = caption
      axis[i].imshow(img)
      axis[i].set_title(text,fontsize=9)
      axis[i].axis('off')
    plt.tight_layout()
    plt.show()




