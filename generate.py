import torch
from model import ffn
import argparse

# todo save character list and don't load it every time
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

def generate(model, prompt):
    inp = torch.zeros(1, len(prompt)).long()
    inp[0,:] = torch.LongTensor(encode(prompt))
    seq = model.generate(inp).squeeze().tolist()
    string = decode(seq)
    string = ' '.join(string)
    print(string)

def main(cfg):
    model = ffn(cfg.vocab_size, cfg.context_length) 
    model.load_state_dict(torch.load(cfg.weights_path))
    model.eval()
    generate(model, cfg.prompt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=65) 
    parser.add_argument('--hidden_size', type=int, default=200) 
    parser.add_argument('--context_length', type=int, default=200)
    parser.add_argument('--weights_path', type=str, default='ckpt10k.pth')
    parser.add_argument('--prompt', type=str, default='When')

    cfg = parser.parse_args()
    main(cfg)

