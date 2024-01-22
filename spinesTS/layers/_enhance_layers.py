import torch
from torch import nn, fft
from torch.nn import functional as F


class GaussianNoise1d(nn.Module):
    """Adds Gaussian noise to the input tensor.

    Parameters
    ----------
    level: float, the scaling multiple of gaussian noise

    Returns
    -------
    torch.Tensor
    """

    def __init__(self, level=0.1):
        super(GaussianNoise1d, self).__init__()
        self.level = level

    def forward(self, x):
        return x + torch.mul(torch.randn_like(x, device=x.device), 1 / torch.median(x) * self.level)


class Time2Vec(nn.Module):
    """Time2Vec module.

    Parameters
    ----------
    in_features: int, input-tensor last dimension shape

    Returns
    -------
    None
    """

    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, 1))
        self.P = nn.Parameter(torch.randn(1))
        self.sin_w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.sin_p = nn.Parameter(torch.randn(out_features-1))
        self.cos_w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.cos_p = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, x):
        original_sin = torch.sin(torch.matmul(x, self.sin_w) + self.sin_p)
        original_cos = torch.cos(torch.matmul(x, self.cos_w) + self.cos_p)

        x = torch.matmul(x, self.W) + self.P

        # last dimension shape (-1, 2 * out_features - 1)
        return torch.concat((original_sin, original_cos, x), dim=-1)


class GAU(nn.Module):
    def __init__(
            self,
            in_features,
            query_key_dim=256,
            expansion_factor=3.,
            skip_connect=True,
            dropout=0.,
    ):
        super(GAU, self).__init__()
        hidden_dim = int(expansion_factor) * in_features

        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.SiLU()
        )

        self.to_qk = nn.Sequential(
            nn.Linear(in_features, query_key_dim),
            nn.SiLU()
        )

        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, in_features),
            nn.Dropout(dropout)
        )

        self.add_residual = skip_connect

    def forward(self, x):
        seq_len = x.shape[-2]

        normed_x = self.norm(x)  # (bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim=-1)  # (bs,seq_len,seq_len)

        Z = self.to_qk(normed_x)  # (bs,seq_len,query_key_dim)

        QK = torch.einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)

        if x.ndim == 2:
            sim = torch.einsum('i d, j d -> i j', q, k) / seq_len
        else:
            sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len

        A = F.relu(sim) ** 2
        A = self.dropout(A)

        if x.ndim == 2:
            V = torch.einsum('i j, j d -> i d', A, v)
        else:
            V = torch.einsum('b i j, b j d -> b i d', A, v)

        V = V * gate

        out = self.to_out(V)

        if self.add_residual:
            out = out + x

        return out


class FFTTopKBlock(torch.nn.Module):
    """
    对传入的二维数组的每一行，进行快速傅里叶变换，并提取Top K个频率和幅值
    """

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def forward(self, x):
        # 假设你的时间序列数据存储在一个torch.Tensor对象中，命名为data
        # data的维度为[batch_size, seq_length]

        # 对时间序列执行FFT
        fft_result = fft.fft(x)

        # 计算FFT的幅值
        amplitude = torch.abs(fft_result)

        # 根据FFT的幅值排序并选择最具代表性的前K个趋势
        top_k_amplitude, _ = torch.topk(amplitude, k=self.k, dim=1)

        # 获取对应的频率信息
        freqs = fft.fftfreq(x.shape[1])
        top_k_freqs, _ = torch.topk(freqs, k=self.k)
        # 频率
        top_k_freqs = top_k_freqs.repeat(x.shape[0], 1)

        return torch.concat((top_k_amplitude, top_k_freqs), dim=1)
