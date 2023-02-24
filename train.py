import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import time
from model import ffn
from data import Data

def train(model, data, optimizer, cfg):

    model.train()
    start = time.time()
    iter_times = []
    for n in range(cfg.num_iters):
        t_s = time.time()
        inp, tgt = data.get_batch(cfg.batch_size, 'train')
        inp = F.one_hot(inp, num_classes=data.vocab_size).float()
        tgt = F.one_hot(tgt, num_classes=data.vocab_size).float()

        logits = model(inp)
        loss = F.cross_entropy(logits, tgt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        iter_times.append(time.time() - t_s)
        if n % cfg.eval_iter == 0:
            print(f'Step: {n}')
            print(f'Loss: {loss}')
            print(f'Avg time per step: {np.mean(iter_times)}')
            iter_times = []

            # validation
            with torch.no_grad():
                inp, tgt = data.get_batch(cfg.batch_size, 'test')
                inp = F.one_hot(inp, num_classes=data.vocab_size).to(dtype=torch.float32)
                tgt = F.one_hot(tgt, num_classes=data.vocab_size).to(dtype=torch.float32)
                logits = model(inp)
                loss = F.cross_entropy(logits, tgt)
                print(f'Validation loss: {loss}')
                torch.save(model.state_dict(), f'checkpoints/{cfg.name}{n}.pth')

    print(f'Total time taken: {time.time() - start}')
    torch.save(model.state_dict(), f'{cfg.name}.pth') 

def main(cfg):
    model = ffn(cfg.vocab_size, cfg.hidden_size)
    data = Data()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train(model, data, optimizer, cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=65) 
    parser.add_argument('--hidden_size', type=int, default=100) 
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--num_iters', type=int, default=20000) 
    parser.add_argument('--eval_iter', type=int, default=1000) 
    parser.add_argument('--name', type=str, default='ckpt')
    cfg = parser.parse_args()
    main(cfg)
