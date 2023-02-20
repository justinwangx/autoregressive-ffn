import torch
import torch.nn.functional as F
from model import ffn
from data import Data
import argparse

def generate(model, data, prompt, max_length=150):
    initial = prompt = prompt[-1]
    prompt = torch.tensor(data.cti[prompt])
    prompt = F.one_hot(prompt, num_classes=cfg.vocab_size).to(dtype=torch.float32)

    output = []
    while len(output) < max_length:
        y = model(prompt)
        idx = torch.argmax(torch.softmax(y, dim=-1)).item()
        print(f'Pred idx: {idx}')
        nc = data.itc[idx]
        print(f'Pred char: {nc}')
        output.append(nc)
        prompt = y

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
    parser.add_argument('--vocab_size', type=int, default=64) 
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--weights_path', type=str, default='ckpt.pth')
    parser.add_argument('--prompt', type=str, default='When')

    cfg = parser.parse_args()
    main(cfg)
