import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class G2P(nn.Module): 
  #TODO beam search 구현
  def __init__(self, encoder, decoder):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder

  def forward(self, src, trg):

    encoder_outputs = self.encoder(src)
    output = self.decoder(trg, encoder_outputs, src=src)
    return output
  
