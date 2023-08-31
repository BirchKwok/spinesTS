import torch
from torch import nn
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

    def __init__(self, in_features):
        super(Time2Vec, self).__init__()

        self.W = nn.Parameter(torch.randn(in_features, in_features))
        self.P = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(1))
        self.p = nn.Parameter(torch.randn(1))
        self.sine_w = nn.Parameter(torch.randn(1))
        self.cosine_w = nn.Parameter(torch.randn(1))

    def forward(self, x):
        original = self.w * x + self.p
        x = torch.matmul(x, self.W)
        for i in range(x.shape[0]):
            x[i, :] = self.sine_w * torch.sin(torch.squeeze(x[i]) + self.P) + \
                      self.cosine_w * torch.cos(torch.squeeze(x[i]) + self.P)

        return torch.concat((original, x), dim=-1)  # last dimension shape (, 2 * in_features)


class GAU(nn.Module):
    def __init__(
            self,
            in_features,
            query_key_dim=128,
            expansion_factor=2.,
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
