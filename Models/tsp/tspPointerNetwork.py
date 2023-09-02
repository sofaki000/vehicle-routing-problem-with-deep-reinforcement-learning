import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

from Models.embeddings.ConvolutionalGraphEmbedding import ConvGraphEmbedding


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau'):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))


    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len    = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query  = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2) #[batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.convEmbedding = ConvGraphEmbedding(input_size, embedding_size)

        # self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        # self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)

        output = self.convEmbedding(inputs.transpose(2,1)).transpose(1,2)
        # [bs, hidden_size, seq_len]

        assert output.size(1) == seq_len
        assert output.size(0) == batch_size
        return output
        # embedding = self.embedding.repeat(batch_size, 1, 1)
        # embedded = []
        # inputs = inputs.unsqueeze(1)
        # for i in range(seq_len):
        #     embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        # embedded = torch.cat(embedded, 1)
        # return embedded

"""<h3>Pointer Network</h3>
<p><a href="https://arxiv.org/abs/1506.03134">Pointer Networks
</a></p>
<p>Check my tutorial - <a href="https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Intro%20to%20Pointer%20Network.ipynb">intro to Pointer Networks</a></p>
<p>The model solves the problem of variable size output dictionaries using a recently proposed mechanism of neural attention. It differs from the previous attention attempts in that, instead of using attention to blend hidden units of an encoder to a context vector at each decoder step, it uses attention as a pointer to select a member of the input sequence as the output.</p>
<img src="imgs/Снимок экрана 2017-12-26 в 4.30.58 ДП.png">
"""

class PointerNet(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            embedding="Graph"):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len

        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf

        return logits, clone_mask

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        #assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)


        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()
        tour_idx = []
        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            #mask.eq(1).all().item()
            # if not ~mask.byte().any()
            #     break

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)


            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)


            idxs = probs.multinomial(1).squeeze(1)

            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break

            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            #tour_idx.append(idxs.data.unsqueeze(1))
            prev_probs.append(probs)
            prev_idxs.append(idxs)


        #tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)

        return prev_probs, prev_idxs

"""<h3>Optimization with policy gradients</h3>
<p>Model-free policy-based Reinforcement Learning to optimize the parameters of a pointer network using the well-known REINFORCE algorithm</p>
"""

class CombinatorialRLTSP(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            embedding,
            attention):
        super(CombinatorialRLTSP, self).__init__()
        self.reward = reward

        self.actor = PointerNet(
                embedding_size,
                hidden_size,
                seq_len,
                n_glimpses,
                tanh_exploration,
                use_tanh,
                attention,
                embedding)


    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len    = inputs.size(2)

        probs, action_idxs = self.actor(inputs)

        actions = []
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])


        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])


        R = self.reward(actions)

        return R, action_probs, actions, action_idxs