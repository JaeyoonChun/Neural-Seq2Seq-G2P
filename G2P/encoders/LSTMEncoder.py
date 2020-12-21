import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LSTMEncoder(nn.Module):
  def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
    super().__init__()

    self.enc_hid_dim = enc_hid_dim
    self.n_layers = n_layers
    self.embedding = nn.Embedding(input_dim, emb_dim)
    
    self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
    self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
    self.dropout = nn.Dropout(dropout)
  
  @classmethod
  def from_args(cls, args, fields, device):
    return cls(
      len(fields['grapheme'].vocab),
      args.enc_emb_dim,
      args.enc_hid_dim,
      args.dec_hid_dim,
      args.n_layers,
      args.enc_dropout
    )
    
  def forward(self, src):
    # src = [src len, batch size]

    embedded = self.dropout(self.embedding(src))
    # embedded = [src len, batch size, emb dim]

    outputs, (hidden, cell) = self.rnn(embedded)
    # outputs = [src len, barch size, hid dim * 2(bidirectional)]
    # hidden = [n layers*2, batch size, hid dim]
    # cell = [n layers*2, batch size, hid dim]

    # outputs are always from the top hidden layer
    # hidden and cell are stacked [forward_1, backward_1, forward_2, backward_2...]
    # hidden [-2, :, : ] is the last of the forwards RNN 
    # hidden [-1, :, : ] is the last of the backwards RNN

    hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
    cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
    # hidden = [batch size, dec hid dim]
    # cell = [batch size, dec hid dim]

    return outputs, hidden, cell    