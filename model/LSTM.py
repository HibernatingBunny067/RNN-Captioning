import torch
import torch.nn as nn 
import torch.nn.functional as F
import random

"""
BahdanauLSTM(): primary model class, inherits from PyTorch's nn.Module class 
- used to initialize a model of the LSTM architecture with single layer
- embedding dim, feature dim of the feature maps, dropout in decoder layer and vocab dim are some accepted parameters 
- vocab is the object of the Vocabulary( ) class 
"""

class BahdanauLSTM(nn.Module):
    def __init__(self,vocab,feature_dim,embedding_dim=256,hidden_dim=512,dropout=0.3,device='cuda'):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.__len__()
        self.L, self.D = feature_dim 
        self.init_h0 = nn.Linear(self.D,hidden_dim) 
        self.init_c0 = nn.Linear(self.D,hidden_dim)
        self.W_h = nn.Linear(hidden_dim,hidden_dim)
        self.W_a = nn.Linear(self.D,hidden_dim)
        self.v = nn.Linear(hidden_dim,1)
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(self.vocab_size,embedding_dim)
        self.LSTM = nn.LSTM(
            input_size=embedding_dim+self.D,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim,self.vocab_size)
        self.start_idx = vocab.word2idx['<start>']
        self.end_idx = vocab.word2idx['<end>']
        self.device = device

    def decoder(self,rnn_output):
        output = self.output_layer(self.dropout(rnn_output))
        return output
        
    def init_hidden(self,image_features):
      N,L,D = image_features.shape
      avg_img_features = torch.sum(image_features,dim=1)/L
      h0 = self.init_h0(avg_img_features)
      c0 = self.init_c0(avg_img_features)
      h0 = h0.unsqueeze(0)
      c0 = c0.unsqueeze(0)
      return h0,c0
    
    def forward(self,image_features,captions,max_len=20):
      '''
      INPUTS:
      image_features: (batch_size,L,D)
      captions: (batch_size,T)

      COMPUTATION: 
      hidden: ((1,batch_size,hidden_dim),(1,batch_size,hidden_dim))
      outputs: (batch,T,vocab_size)
      context: (batch_size,D)
      current_input: (batch_size,hidden_dim,embed+D)
      output: (batch_size,1,hidden_dim)
      vocab_scores: (batch_size,vocab_dim)

      OUTPUT:
      ouptuts: (batch_size, T, vocab_dim)
      '''
      batch,T = captions.shape
      hidden = self.init_hidden(image_features)
      outputs = torch.zeros(batch,T,self.vocab_size).to(self.device)

      for t in range(T): #Teacher forcing 
          context,_ = self.attention(hidden[0].squeeze(0),image_features)
          current_input = torch.cat([self.embeddings(captions[:,t]).unsqueeze(1),context],dim=2)
          output, hidden = self.LSTM(current_input,hidden) 
          vocab_scores = self.decoder(output.squeeze(1))
          outputs[:,t,:] = vocab_scores # directly returning the vocab scores 
      return outputs


    def attention(self,decoder_hidden,encoder_outputs): # additive attention using tanh( ) function
        '''
        decoder_hidden: (N,H) {(N,1,H) -> (N,L,H)}
        ecoder_ouput: (N,L,D)
        context vector:  (N,D)
        '''
        N,L,D = encoder_outputs.shape
        h = decoder_hidden.unsqueeze(1).repeat(1,L,1) #(N,L,H)
        encoded_proj = self.W_a(encoder_outputs) #(N,L,H)
        hidden_proj = self.W_h(h) #(N,L,H)
        energy = torch.tanh(encoded_proj+hidden_proj) #(N,L,H)
        scores = self.v(energy).squeeze(2) #(N,L)
        attn_weights = F.softmax(scores,dim=1) #(N,L)
        context = torch.bmm(attn_weights.unsqueeze(1),encoder_outputs) #(N,1,D)
        return context,attn_weights#(N,D)
    
    @torch.no_grad() # do I need to pass captions in validation ? NOPE !!!
    def generate(self, image_features,max_len=20,greedy=True):
      self.eval()
      batch_size = image_features.shape[0]
      hidden = self.init_hidden(image_features)
      inputs = torch.full((batch_size,), self.start_idx, dtype=torch.long).to(self.device) #batch of tensor with <start> token 
      generated = torch.zeros((batch_size,max_len),dtype=torch.long,device=self.device)
      predicted = None
      for t in range(max_len):
          context,_ = self.attention(hidden[0].squeeze(0), image_features)           
          emb = self.embeddings(inputs).unsqueeze(1)                     
          current_input = torch.cat([emb, context], dim=2)
          output, hidden = self.LSTM(current_input, hidden)             
          logits = self.decoder(output.squeeze(1))       
          if greedy:                 
            predicted = logits.argmax(dim=1)                       
          else:
            probs = F.softmax(logits,dim=1)
            predicted = torch.multinomial(probs,num_samples=1).squeeze(1)
          generated[:, t] = predicted                              
          inputs = predicted
          if (inputs == self.end_idx).all():
            break
      return generated ##gives the token id of the words 

    @torch.no_grad() 
    def generate_with_maps(self, image_features,max_len=20,greedy=True):
      self.eval()
      batch_size = image_features.shape[0]
      hidden = self.init_hidden(image_features)
      inputs = torch.full((batch_size,), self.start_idx, dtype=torch.long).to(self.device) #batch of tensor with <start> token 
      generated = torch.zeros((batch_size,max_len),dtype=torch.long,device=self.device)
      predicted = None
      context_vectors = []
      for t in range(max_len):
          context,attn_weights = self.attention(hidden[0].squeeze(0), image_features)
          context_vectors.append(attn_weights)           
          emb = self.embeddings(inputs).unsqueeze(1)                     
          current_input = torch.cat([emb, context], dim=2)
          output, hidden = self.LSTM(current_input, hidden)             
          logits = self.decoder(output.squeeze(1))       
          if greedy:                 
            predicted = logits.argmax(dim=1)                       
          else:
            probs = F.softmax(logits,dim=1)
            predicted = torch.multinomial(probs,num_samples=1).squeeze(1)
          generated[:, t] = predicted                              
          inputs = predicted
          if (inputs == self.end_idx).all():
            break
      return generated,context_vectors ##