from torch import nn
import torch

class ConvGraphEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ConvGraphEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    def forward(self, inputs):
        batch_size = inputs.size(0)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(batch_size):
            el = self.embedding(torch.transpose(inputs[i, 0], 0, 1))
            embedded.append(torch.unsqueeze(el, 0))
        embedded = torch.cat(embedded,0)
        return embedded