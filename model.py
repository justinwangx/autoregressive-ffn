import torch
import torch.nn as nn

class ffn(nn.Module):
    def __init__(self, vocab_size, context_length):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, 200)
        self.linear = nn.Linear(200, 200)
        self.proj = nn.Linear(200, vocab_size)
        self.relu = nn.ReLU()
        self.context_length = context_length

    def forward(self, x, targets=None):
        logits = self.proj(self.relu(self.linear(self.embd(x))))
        return logits
    
    def generate(self, seq, max_new_chars=150):
        for _ in range(max_new_chars):
            context = seq if seq.size(1) <= self.context_length else seq[:,-self.context_length:]
            logits = self(context)
            logits = logits[:,-1,:] # .unsqueeze(1)

            v, _ = torch.topk(logits, min(3, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

            pred = F.softmax(logits, dim=-1)
            pred = torch.multinomial(pred, num_samples=1)
            seq = torch.cat([seq, pred], dim=1)
        return seq