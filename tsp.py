import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
class TSPDataset(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]
    def reward(self,sample_solution):
        """
        sample_solution seq_len of [batch_size]
        """
        batch_size = sample_solution[0].size(0)
        n = len(sample_solution)
        tour_len = Variable(torch.zeros([batch_size]))

        for i in range(n - 1):
            tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)

        tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

        return tour_len
    def update_mask(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 0 #1

        return clone_mask
    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf

        return logits, clone_mask

class TSPDatasetGaussian(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDatasetGaussian, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            m = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
            x = m.sample(sample_shape=(2,num_nodes))
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]



def reward(sample_solution):
    """
    sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))

    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)

    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

    return tour_len
