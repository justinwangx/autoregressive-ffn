import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from model import ffn

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

def get_batch(batch_size, split='train'):
    """
    Returns a LongTensor of shape (bs, context_length) 
    """
    data = train_data if split=='train' else test_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([torch.LongTensor(train_data[i:i+context_length]) for i in ix])
    y = torch.stack([torch.LongTensor(train_data[i+1:i+1+context_length]) for i in ix])
    return x, y

def train(model, optimizer, cfg):

    model.train()
    for n in range(cfg.num_iters):
        inp, tgt = get_batch(cfg.batch_size, 'train')
        logits = model(inp)
        tgt = tgt.long()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if cfg.num_iters % cfg.eval_iter == 0:
            print(f'Step: {n}')
            print(f'Loss: {loss}')
    
    torch.save(model.state_dict(), cfg.save_path) 

def main(cfg):
    model = ffn()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    train(model, optimizer, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser
    parser.add_argument('--vocab_size', type=int, default=65) 
    parser.add_argument('--hidden_size', type=int, default=200) 
    parser.add_argument('--context_length', type=int, default=200) 
    parser.add_argument('--batch_size', type=int, default=100) 
    parser.add_argument('--num_iters', type=int, default=10000) 
    parser.add_argument('--eval_iters', type=int, default=100) 
    parser.add_argument('--save_path', type=str, default='./ckpt.pth')
    cfg = parser.parse_args()
    main(cfg)

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