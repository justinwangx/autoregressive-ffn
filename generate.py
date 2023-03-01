import torch
import torch.nn.functional as F
from model import ffn
from data import Data
import argparse

def generate(model, data, prompt, max_length=150):
    prompt = prompt[-data.context_length:]
    # pad with spaces
    if len(prompt) < data.context_length:
        prompt = ' ' * (data.context_length - len(prompt)) + prompt
    output = [c for c in prompt]

    prompt = torch.tensor([data.cti[ch] for ch in prompt])
    prompt = F.one_hot(prompt, num_classes=data.vocab_size).float().view(1, -1)

    while len(output) < max_length:
        y = model(prompt)
        pred = torch.softmax(y, dim=-1)
        idx = torch.multinomial(pred, num_samples=1).item()
        # idx = torch.argmax(pred, dim=-1).item()
        nc = data.itc[idx]
        output.append(nc)

        nc_one_hot = F.one_hot(torch.tensor(idx), num_classes=data.vocab_size).float().view(1, -1)
        prompt = torch.cat([prompt[:, data.vocab_size:], nc_one_hot], dim=1)

    output = ''.join([c for c in output])
    print(output)

def main(cfg):
    data = Data(cfg.context_length)
    model = ffn(data.vocab_size, cfg.hidden_size, cfg.context_length) 
    model.load_state_dict(torch.load(cfg.weights_path))
    model.eval()
    generate(model, data, cfg.prompt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--context_length', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--weights_path', type=str, default='checkpoints/ckpt17000.pth')
    parser.add_argument('--prompt', type=str, default='Hello')

    cfg = parser.parse_args()
    main(cfg)
