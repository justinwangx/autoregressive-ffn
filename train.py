import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# data processing
# all characters (1,115,394 total)
data = open('data/tinyshakespeare.txt', 'r').read()
# unique characters (65 total)
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
cti = {c:i for i,c in enumerate(chars)}
itc = {i:c for i,c in enumerate(chars)}

def encode(chars):
    return [cti[c] for c in chars]
def decode(nums):
    return [itc[n] for n in nums]

# create train and test splits
split = int(0.9*len(data))
train_data = encode(data[:split])
test_data = encode(data[split:])

# hyperparams
context_length = 200
batch_size = 100
num_iters = 10000
eval_iter = 100

def get_batch(split='train'):
    """
    Returns a LongTensor of shape (bs, context_length) 
    """
    data = train_data if split=='train' else test_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.LongTensor(train_data[i:i+context_length]) for i in ix])
    y = torch.stack([torch.LongTensor(train_data[i+1:i+1+context_length]) for i in ix])
    return x, y

class ffn(nn.Module):
    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, 200)
        self.linear = nn.Linear(200, 200)
        self.proj = nn.Linear(200, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x, targets=None):
        logits = self.proj(self.relu(self.linear(self.embd(x))))
        return logits
    
    def generate(self, seq, max_new_chars=150):
        for _ in range(max_new_chars):
            context = seq if seq.size(1) <= context_length else seq[:,-context_length:]
            logits = self(context)
            logits = logits[:,-1,:] # .unsqueeze(1)

            v, _ = torch.topk(logits, min(3, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

            pred = F.softmax(logits, dim=-1)
            pred = torch.multinomial(pred, num_samples=1)
            seq = torch.cat([seq, pred], dim=1)
        return seq

def train(model, optimizer):
    model.train()
    for n in range(num_iters):
        inp, tgt = get_batch('train')
        logits = model(inp)
        tgt = tgt.long()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % eval_iter == 0:
            print(f'Step: {n}')
            print(f'Loss: {loss}')
    
    torch.save(model.state_dict(), 'ckpt10k.pth') # final loss: 2.49

if __name__ == '__main__':
    # model = ffn()
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)

    if False: 
        train(model, optimizer)
    
    # model.load_state_dict(torch.load('ckpt10k.pth'))
    # model.eval()
    # prompt = torch.zeros(1, 3).long()
    # prompt[0, :] = torch.LongTensor(encode('Our'))
    # seq = model.generate(prompt).squeeze().tolist()
    # string = decode(seq)
    # string = ' '.join(string)
    # print(string)

    # checking input output pairs
    context_length = 10
    x,y = get_batch()
    for i in range(3):
        inp = ' '.join(decode(x[i].tolist()))
        out = ' '.join(decode(y[i].tolist()))
        print(f'y{i}: {out}')
        print(f'x{i}: {inp}')
        print()