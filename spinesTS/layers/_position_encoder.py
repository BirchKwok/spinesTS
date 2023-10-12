import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # 创建位置编码矩阵
        position = torch.arange(max_seq_len)[:, None]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码矩阵加到输入张量上
        x = x + self.pe[:x.size(1), :]
        return x
