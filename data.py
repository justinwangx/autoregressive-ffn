import torch
import random
import re

class Data():
    def __init__(self):
        # all characters (0,115,394 total)
        # unique characters (65 total)
        self.data = open('data/tinyshakespeare.txt', 'r').read()
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)
        self.cti = {c:i for i,c in enumerate(self.chars)}
        self.itc = {i:c for i,c in enumerate(self.chars)}

        # split data into sentences
        # self.data = re.split('r[.?! \n]+', self.data)
        self.data = self.data.splitlines()
        while '' in self.data:
            self.data.remove('')
        # create train and test splits
        split = int(0.9*len(self.data))
        self.train_data = self.data[:split]
        self.test_data = self.data[split:]
    
    def get_batch(self, batch_size, split='train'):
        data = self.train_data if split=='train' else self.test_data
        # next-character pairs
        x = []
        y = []
        # x = torch.tensor(batch_size, 1)
        # y = torch.tensor(batch_size, 1)
        for i in range(batch_size):
            sentence = random.choice(data) + '\n'

            # sentence = data[0] # check to overfit on a sentence
            while len(sentence) < 2:
                sentence = random.choice(data)
            idx = random.randint(0, len(sentence)-2)
            x.append(self.cti[sentence[idx]])
            y.append(self.cti[sentence[idx+1]])
        return torch.tensor(x), torch.tensor(y)

    def encode(self, chars):
        return [self.cti[c] for c in chars]

    def decode(self, nums):
        return [self.itc[n] for n in nums]
