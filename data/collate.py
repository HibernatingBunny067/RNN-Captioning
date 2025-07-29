import torch
from torch.nn.utils.rnn import pad_sequence 
'''
Function used to pad caption sequences to the longest caption per batch during data loading 
'''
class Padding():
	def __init__(self,pad_idx):
		self.pad_idx = pad_idx

	def __call__(self,batch):
		images, captions, ids= zip(*batch)
		images = torch.stack(images,dim=0)
		padded_captions = pad_sequence(captions,batch_first=True,padding_value=self.pad_idx)
		return images,padded_captions,ids