import torch.nn as nn


class SignGRU(nn.Module):
    def __init__(self, classes, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(225, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        _, state = self.gru(x)
        return self.head(state[-1])
