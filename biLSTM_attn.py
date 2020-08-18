import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
    super().__init__()

    self.enc_hid_dim = enc_hid_dim
    self.n_layers = n_layers
    self.embedding = nn.Embedding(input_dim, emb_dim)
    self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True)
    self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
    self.dropout = nn.Dropout(dropout)
  
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
    cell = torch.tanh(self.fc(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1))) # dim=1 이 차원을 아예 없애나?
    # hidden = [batch size, dec hid dim]
    # cell = [batch size, dec hid dim]

    return outputs, hidden, cell    

class Attention(nn.Module):
  def __init__(self, enc_hid_dim, dec_hid_dim):
    super().__init__()
    self.attn = nn.Linear((enc_hid_dim*2)+dec_hid_dim, dec_hid_dim) # 바다나우 어텐션, concat
    self.v = nn.Linear(dec_hid_dim, 1, bias=False)
  
  def forward(self, hidden, encoder_outputs):
  
    # hidden = [batch size, dec hid dim]
    # encoder_outputs = [src len, batch size, enc hid dim*2]

    batch_size = encoder_outputs.shape[1]
    src_len = encoder_outputs.shape[0]

    # repeat decoder hidden state src_len times
    hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    # hidden = [batchsize, src len, dec hid dim]
    # encoder_outputs = [batch size, src len, enc hid dim*2]

    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
    # energy = [batch size, src len, dec hid dim]

    attention = self.v(energy).squeeze(2)
    # attention = [batch size, src len]

    return F.softmax(attention, dim=1)

class Decoder(nn.Module):
  def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, attention, n_layers, dropout):
    super().__init__()

    self.output_dim = output_dim
    self.dec_hid_dim = dec_hid_dim
    self.attention = attention
    self.n_layers = n_layers

    self.embedding = nn.Embedding(output_dim, emb_dim)
    self.rnn = nn.LSTM((enc_hid_dim*2)+emb_dim, dec_hid_dim, n_layers, dropout=dropout)
    self.fc_out = nn.Linear((enc_hid_dim*2)+emb_dim+dec_hid_dim, output_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, input, hidden, cell, encoder_outputs):
    # input = [batch size]
    # hidden = [batch size, dec hid dim]
    # cell = [batch size, dec hid dim]
    # encoder_outputs = [src len, batch size, enc hid dim*2]
    input = input.unsqueeze(0)
    # input = [1, batch size]

    embedded = self.dropout(self.embedding(input))
    # embedded = [1, batch size, emb dim]

    a = self.attention(hidden, encoder_outputs)
    # a = [batch size, src len]

    a = a.unsqueeze(1)
    # a = [batch size, 1, src len]

    encoder_outputs = encoder_outputs.permute(1, 0, 2)
    # encoder_outputs = [batch size, src len, enc hid dim*2]

    weighted = torch.bmm(a, encoder_outputs)
    # weighted = [batch size, 1, enc hid dim*2]
    # bmm : 맨 앞 batch 차원은 유지하면서 뒤에 요소들의 행렬곱

    weighted = weighted.permute(1, 0, 2)
    # weighted = [1, batch size, enc hid dim*2]
    
    rnn_input = torch.cat((embedded, weighted), dim=2)
    # rnn_input = [1, batch size, (enc hid dim*2)+emb dim]

    output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
    #output = [seq len, batch size, hid dim * n directions]
    #hidden = [n layers * n directions, batch size, hid dim]
    #cell = [n layers * n directions, batch size, hid dim]
    
    #seq len and n directions will always be 1 in the decoder, therefore:
    #output = [1, batch size, hid dim]
    #hidden = [n layers, batch size, hid dim]
    #cell = [n layers, batch size, hid dim]

    embedded = embedded.squeeze(0)
    output = output.squeeze(0)
    weighted = weighted.squeeze(0)

    prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
    # prediction = [batch size, output dim]

    return prediction, hidden.squeeze(0), cell.squeeze(0)

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.device = device

  def forward(self, src, trg, teacher_forcing_ratio=0.5):
    # src = [src len, batch size]
    # trg = [trg len, batch size]
    batch_size = trg.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim

    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    encoder_outputs, hidden, cell = self.encoder(src)

    input = trg[0, :]
    for t in range(1, trg_len):

      output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1)
      input = trg[t] if teacher_force else top1
    
    return outputs
