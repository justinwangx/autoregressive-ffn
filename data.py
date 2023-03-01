import numpy as np
import torch

class Data():
    def __init__(self, context_length):
        # all characters (0,115,394 total)
        # unique characters (65 total)
        self.data = open('data/tinyshakespeare.txt', 'r').read()
        self.chars = sorted(list(set(self.data)))
        self.vocab_size = len(self.chars)
        self.cti = {c:i for i,c in enumerate(self.chars)}
        self.itc = {i:c for i,c in enumerate(self.chars)}
        self.context_length = context_length

        # convert data to np array of ints
        data = []
        for ch in self.data:
            data.append(self.cti[ch])
        self.data = np.array(data)

        # create dataset of character ngrams, train and test splits
        split = int(0.9 * self.data.shape[0])
        strides = self.data.strides[0]
        self.data = np.lib.stride_tricks.as_strided(data, shape=(self.data.shape[0]-context_length, context_length+1), strides=(strides, strides))
        X, Y = self.data[:, :context_length], self.data[:, context_length]
        self.Xtrain, self.Xtest = X[:split], X[split:]
        self.Ytrain, self.Ytest = Y[:split], Y[split:]
        del self.data
    
    def get_batch(self, batch_size, split='train'):
        X, Y = (self.Xtrain, self.Ytrain) if split=='train' else (self.Xtest, self.Ytest)
        # next-character pairs
        batch_indices = np.random.randint(0, X.shape[0], (batch_size,))
        return torch.tensor(X[batch_indices]), torch.tensor(Y[batch_indices])
