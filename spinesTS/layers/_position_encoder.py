import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """1D时间序列位置编码"""
    def __init__(self, d_model, max_len=5000, add_x=True):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.add_x = add_x
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入上
        if self.add_x:
            x = x + self.pe[:, :x.size(1)]
            return x
        return self.pe[:, :x.size(1)]
