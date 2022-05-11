from torch import nn


class SeriesRecombinationLayer(nn.Module):
    def __init__(self, in_shapes, out_features=128, rnn_layers=1, dropout=0.):
        assert isinstance(in_shapes, tuple) and len(in_shapes) == 2
        super(SeriesRecombinationLayer, self).__init__()
        (self.rows, self.cols), self.out_features = in_shapes, out_features

        self.encoder_rnn = nn.GRU(self.rows, out_features, batch_first=True, num_layers=rnn_layers,
                                  dropout=dropout if rnn_layers > 1 else 0.)
        self.decoder_rnn = nn.GRU(out_features, out_features, batch_first=True, num_layers=rnn_layers,
                                  dropout=dropout if rnn_layers > 1 else 0.)

        self.out = nn.Linear(self.cols * out_features, out_features)

    def forward(self, x):
        assert x.ndim == 3

        x = x.permute((0, 2, 1))
        x, h = self.encoder_rnn(x)
        x, h = self.decoder_rnn(x, h)
        x = x.permute((0, 2, 1))

        return self.out(x.reshape((-1, self.cols * self.out_features)))  # (batch_size, out_features)

