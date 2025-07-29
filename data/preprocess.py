import json
import os
from PIL import Image
import random
import re
from collections import Counter
import torch
from torch.utils.data import Dataset
from collections import Counter
import json
import json
import random
from collections import defaultdict

'''
Vocabulary(): primary class to preprocess the captions and make a vocabulary of words based on min_freq and max_freq of words and filtered by min and max caption length 

KarpathySplit(): class used to process the KarpathySplit file of the MS COCO dataset 
- used to get a shuffled subset of the data in three splits train,val and test based on user input fractions

'''


class Vocabulary():
    def __init__(self, min_freq=5,max_size=None,max_len=25,min_len=8):
        self.word2idx = {}
        self.idx2word ={}
        self.freqs = Counter()
        self.min_freq = min_freq
        self.max_size = max_size
        self.max_len = max_len
        self.min_len = min_len
        self.list_of_dicts = []
        self.SPECIALS = ['<pad>', '<start>', '<end>', '<unk>']
        for i, tokens in enumerate(self.SPECIALS):
            self.word2idx[tokens] = i
            self.idx2word[i] = tokens
        self.next_index = len(self.SPECIALS)

    def tokenize(self,sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^\w\s]", "", sentence)
        return sentence.split()

    def build(self,captions):#captions will now be a list of dictionaries 
        for dicts in captions:
            for caps in dicts['captions']:
              token = self.tokenize(caps)
              self.freqs.update(token)

        for word, freq in self.freqs.most_common():
            if freq < self.min_freq:
                continue
            if word not in self.word2idx:
                self.word2idx[word] = self.next_index
                self.idx2word[self.next_index] = word
                self.next_index += 1
                if self.max_size and self.next_index >= self.max_size:
                    break
    def encode(self, caption):
        tokens = self.tokenize(caption)
        tokens = ['<start>']  + tokens + ['<end>']
        return [self.word2idx.get(token,self.word2idx['<unk>']) for token in tokens]

    def decode(self,list_of_idx,with_specials = False):
        word = [self.idx2word.get(index,'<unk>') for index in list_of_idx]
        output = ' '.join(word)
        if with_specials:
            return output
        else:
            return output.replace('<start> ', '').replace(' <end>', '').replace('<pad>','')
    def __len__(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'min_freq': self.min_freq,
                'max_size': self.max_size
            }, f)

    def save_datasets(self,datasets,path):
      trdict,vdict,tdict = [],[],[]
      dicts = [trdict,vdict,tdict]
      for idx,dataset in enumerate(datasets):
        for things in dataset:
          dicts[idx].append(things)
      with open(path,'w') as f:
        json.dump({
          'enc_train':trdict, #list of tuples 
          'enc_val':vdict,
          'enc_test':tdict
        }, f)
        return len(trdict),len(vdict),len(tdict)

    def load(self, path):
      with open(path, 'r') as f:
          data = json.load(f)
          self.word2idx = data['word2idx']
          self.idx2word = {int(v): k for k, v in self.word2idx.items()}
          self.min_freq = data['min_freq']
          self.max_size = data['max_size']
          self.next_index = len(self.word2idx)

    def encode_captions(self,data):
        encoded_data = [] 
        for dicts in data:
          captions = []
          add = {}
          for caps in dicts['captions']:
            token = self.tokenize(str(caps))
            if len(token) > self.min_len and len(token) < self.max_len:
              encoded_caption = self.encode(str(caps))
              captions.append(encoded_caption)
          for things in list(dicts.items()):
            add[things[0]] = things[1]
          add['captions'] = captions
          encoded_data.append(add)
        return encoded_data

    def load_datasets(self, file_path, root_dir='filtered_images'):
        with open(file_path, 'r') as f:
            data = json.load(f)

        enc_train, enc_val, enc_test = [], [], []

        split_map = {
            'enc_train': enc_train,
            'enc_val': enc_val,
            'enc_test': enc_test
        }

        for split, entries in data.items():
            for things in entries:
                filename = things[0]
                encoded_caption = things[1]
                split_name = split.replace('enc_', '')
                image_path = os.path.join(split,filename)
                split_map[split].append((image_path, encoded_caption))

        return enc_train, enc_val, enc_test

        


class KarpathySplit():
    def __init__(self,path,num_train=20000,num_val=5000,num_test=100,seed=42):
        self.karpathy_path = path
        self.num_train=num_train
        self.num_val = num_val
        self.num_test = num_test
        self.splits = ['train','val','test']
        self.nums = [self.num_train,self.num_val,self.num_test]
        random.seed(seed)

    def get_splits(self):
      '''
      return list of dictionaries 
      {'path': img_path,
      'id': img_id,
      'captions';[caption1, caption2,caption3,caption4,caption5] }
      '''
      vocab = Vocabulary()
      with open(self.karpathy_path) as f:
            data = json.load(f)
      split_dicts = {
                        'train':[], 
                       'val':[],
                       'test':[]
                       }
      for idx,splits in enumerate(self.splits):
          images = [img for img in data['images'] if img['split'] == splits] ##get's all the images 
          random.shuffle(images)
          images = images[:self.nums[idx]] ##gets the specific number of images
          for img in images: #for each image
              filename = img['filename'] #takeout the file name
              img_id = img['imgid'] #take out the image id 
              captions = [
                re.sub(r"[^\w\s]", "",  sent['raw'].lower())
                for sent in img['sentences']
                ] #collect all the captions
              datapoint = {
                'path':filename,
                'id':img_id,
                'captions':captions
              }
              split_dicts[splits].append(datapoint)

      return split_dicts['train'],split_dicts['val'],split_dicts['test']