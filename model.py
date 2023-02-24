import torch.nn as nn

class ffn(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, x):
        return self.net(x)
    