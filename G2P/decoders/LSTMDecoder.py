import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from G2P.modules.attention import Attention

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, teacher_forcing_ratio, device):
        super().__init__()

        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)
        self.n_layers = n_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM((enc_hid_dim*2)+emb_dim, dec_hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((enc_hid_dim*2)+emb_dim+dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device
    
    @classmethod
    def from_args(cls, args, fields, device):
        return cls(
            len(fields['phoneme'].vocab),
            args.dec_emb_dim,
            args.enc_hid_dim,
            args.dec_hid_dim,
            args.n_layers,
            args.dec_dropout,
            args.teacher_forcing_ratio,
            device
        )
        
    def forward(self, trg, encoder_outputs, **kwargs):
        enc_src, hidden, cell = encoder_outputs

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
       
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, cell = self._run(input, hidden, cell, enc_src)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs
            
    def _run(self, input, hidden, cell, enc_src):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # cell = [batch size, dec hid dim]
        # enc_src = [src len, batch size, enc hid dim*2]
        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, enc_src)
        # a = [batch size, src len]

        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        enc_src = enc_src.permute(1, 0, 2)
        # enc_src = [batch size, src len, enc hid dim*2]

        weighted = torch.bmm(a, enc_src)
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
