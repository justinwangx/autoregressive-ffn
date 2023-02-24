import torch
import torch.nn.functional as F
from model import ffn
from data import Data
import argparse

def generate(model, data, prompt, max_length=150):
    initial = prompt = prompt[-1]
    prompt = torch.tensor(data.cti[prompt])
    prompt = F.one_hot(prompt, num_classes=cfg.vocab_size).float()

    output = []
    while len(output) < max_length:
        y = model(prompt)
        pred = torch.softmax(y, dim=-1)
        idx = torch.multinomial(pred, num_samples=1).item()
        nc = data.itc[idx]
        output.append(nc)
        prompt = F.one_hot(torch.tensor(idx), num_classes=cfg.vocab_size).float()

    output = ''.join([c for c in output])
    print(f'{initial}{output}')

def main(cfg):
    model = ffn(cfg.vocab_size, cfg.hidden_size) 
    model.load_state_dict(torch.load(cfg.weights_path))
    model.eval()
    data = Data()
    generate(model, data, cfg.prompt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=65) 
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--weights_path', type=str, default='ckpt.pth')
    parser.add_argument('--prompt', type=str, default='When')

    cfg = parser.parse_args()
    main(cfg)
