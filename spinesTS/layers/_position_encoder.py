import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, add_x=True):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        self.add_x = add_x
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.add_x:
            return x + self.pe[:, :x.shape[1]]
        return self.pe[:, :x.shape[1]]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, add_x=True):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.add_x = add_x

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).long().to(x.device)
        if not self.add_x:
            return self.positional_encoding(positions)

        return x + self.positional_encoding(positions)
