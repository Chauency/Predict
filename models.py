import torch.nn as nn


class Model1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Model1, self).__init__()

        self.fc_in = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
            )

        self.fc_out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
            )

        self.lstm1 = nn.LSTMCell(128, 128)
        self.lstm2 = nn.LSTMCell(128, 128)

    def forward(self, inputs):
        seq, (h1, c1), (h2, c2) = inputs
        for i in range(seq.shape[1]):
            x = self.fc_in(seq[:, i, :])
            h1, c1 = self.lstm1(x, (h1, c1))
            h2, c2 = self.lstm2(h1, (h2, c2))
        return self.fc_out(h2)
