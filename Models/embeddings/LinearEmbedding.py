from torch import nn
import torch

class LinearGraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(LinearGraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(batch_size):
            el = self.embedding(torch.transpose(inputs[i, 0], 0, 1))
            embedded.append(torch.unsqueeze(el, 0))
        embedded = torch.cat(embedded,0)
        return embedded