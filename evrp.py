from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EVRPDataset(Dataset):
    def __init__(self, train_size, num_nodes, t_limit, capacity, num_afs=3, seed=520):
        super().__init__()
        self.size = train_size

        torch.manual_seed(seed)
        afs = torch.rand(2,num_afs+1)
        customers = torch.rand(train_size, 2, num_nodes, device=device)

        self.static = torch.cat((afs.unsqueeze(0).repeat(train_size, 1, 1), customers), dim=2).to(device)  # (train_size, 2, num_nodes+4)

        self.dynamic = torch.ones(train_size, 3, 1+num_afs+num_nodes, device=device)   # time duration, capacity, demands
        self.dynamic[:, 0, :] *= t_limit
        self.dynamic[:, 1, :] *= capacity
        self.dynamic[:, 2, :num_afs+1] = 0

        seq_len = self.static.size(2)
        self.distances = torch.zeros(train_size, seq_len, seq_len, device=device)
        for i in range(seq_len):
            self.distances[:, i] = cal_dis(self.static[:, 0, :], self.static[:, 1, :], self.static[:, 0, i:i+1], self.static[:, 1, i:i+1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.static[idx], self.dynamic[idx], self.distances[idx]    # dynamic: None

    def update_mask(self, what):
        pass

    def update_dynamic(self, old_idx, idx, mask, dynamic, distances, dis_by_afs, capacity, velocity, cons_rate, t_limit, num_afs):
        """
        :param old_idx: (batch*beam, 1)
        :param idx: ditto
        :param mask: (batch*beam, seq_len)
        :param dynamic: (batch*beam, dynamic_features, seq_len)
        :param distances: (batch*beam, seq_len, seq_len)
        :param dis_by_afs: (batch*beam, seq_len)
        :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
        :return: updated dynamic AFS = merh pou kanoume recharge
        """

        #TODO: kane th maska apo 1:paw, 0 den paw, se 0 paw, -inf den paw
        # mask[mask == 1] = 0# mporeis na pas
        # mask[mask == 0] =  float('-inf')
        dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx.squeeze(1)].unsqueeze(1)
        depot = idx.eq(0).squeeze(1)
        afs = (idx.gt(0) & idx.le(num_afs)).squeeze(1)
        fs = idx.le(num_afs).squeeze(1)
        customer = idx.gt(num_afs).squeeze(1)  # TODO: introduce num_afs
        time = dynamic[:, 0, :].clone()
        fuel = dynamic[:, 1, :].clone()
        demands = dynamic[:, 2, :].clone()

        time -= dis / velocity
        time[depot] = t_limit #exeis full diathesimh wra
        time[afs] -= 0.25 # katanalwneis ligh wra gia na forthseis
        time[customer] -= 0.5  # katanalwneis ligh wra gia na afhseis to proion

        fuel -= cons_rate * dis
        fuel[fs] = capacity
        demands.scatter_(1, idx, 0)

        dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)
        #print(f'Here are some demands left{((dynamic[:, 2, :] == 0)==False).nonzero()}')
        mask.scatter_(1, idx, float('-inf'))
        # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
        mask[fs, 1:num_afs + 1] = float('-inf')
        mask[afs, 0] = 0
        # kapou edw
        mask[customer, :num_afs + 1] = 0  # phgaine
        mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])
        #print(f'non inf mask {torch.nonzero(mask != float('-inf')).squeeze()}')
        # path1: ->Node->Depot
        dis1 = distances[torch.arange(distances.size(0)), idx.squeeze(1)].clone()
        fuel_pd0 = cons_rate * dis1 #poso fuel xreiazetai na paei apo node sto depot
        time_pd0 = dis1 / velocity # poso time xreiazetai
        dis1[:, num_afs + 1:] += distances[:, 0, num_afs + 1:]
        fuel_pd1 = cons_rate * dis1
        time_pd1 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + distances[:, 0, :]) / velocity
        time_pd1[:, 1:num_afs + 1] += 0.25
        time_pd1[:, num_afs + 1:] += 0.5

        # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
        dis2 = distances[:, 1:num_afs + 1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
        dis2[:, 0] = 0
        dis2 += distances[torch.arange(distances.size(0)), idx.squeeze(1)]
        fuel_pd2 = cons_rate * dis2
        time_pd2 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + dis_by_afs[0]) / velocity
        time_pd2[:, 1:num_afs + 1] += 0.25
        time_pd2[:, num_afs + 1:] += 0.5

        # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
        # the next node should be able to return to depot with at least one way; otherwise, mask it
        mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = float('-inf')
        # edw ginetai h malakia
        mask[(fuel < fuel_pd0) | (time < time_pd0)] = float('-inf')

        all_masked = mask[:, num_afs + 1:].eq(0).sum(1).le(0)
        mask[all_masked, 0] = 0  # unmask the depot if all nodes are masked

        return dynamic

    def update_dynamic2(self, old_idx, idx, mask, dynamic, distances, dis_by_afs, capacity, velocity, cons_rate, t_limit,
                       num_afs):
        """
        1: mporeis na pas
        0: den mporeis na pas
        :param old_idx: (batch*beam, 1)
        :param idx: ditto
        :param mask: (batch*beam, seq_len)
        :param dynamic: (batch*beam, dynamic_features, seq_len)
        :param distances: (batch*beam, seq_len, seq_len)
        :param dis_by_afs: (batch*beam, seq_len)
        :param capacity, velocity, cons_rate, t_limit, num_afs: scalar
        :return: updated dynamic
        """
        dis = distances[torch.arange(distances.size(0)), old_idx.squeeze(1), idx.squeeze(1)].unsqueeze(1)
        episkepthkes_depot = idx.eq(0).squeeze(1)
        afs = (idx.gt(0) & idx.le(num_afs)).squeeze(1) #episkepthkes fulling station
        fs = idx.le(num_afs).squeeze(1) ##episkepthkes fulling station
        customer = idx.gt(num_afs).squeeze(1)  # TODO: introduce num_afs
        time = dynamic[:, 0, :].clone()
        fuel = dynamic[:, 1, :].clone()
        demands = dynamic[:, 2, :].clone()

        time -= dis / velocity
        time[episkepthkes_depot] = t_limit
        time[afs] -= 0.25
        time[customer] -= 0.5

        fuel -= cons_rate * dis # poso kausimo xalases?
        fuel[fs] = capacity # edw episkfthkes fuel station
        #demands.scatter_(1, idx, 1) # mhpws 1?

        print(f'Chose indexes:{idx.squeeze()}')
        demands.scatter_(1, idx, 0) # satisfy the demands of the customer you visit

        # dynamic: [time, fuel, demand]
        dynamic = torch.cat((time.unsqueeze(1), fuel.unsqueeze(1), demands.unsqueeze(1)), dim=1).to(device)

        # mask.scatter_(1, idx, float('-inf'))
        mask.scatter_(1, idx, 0) # MHN XANAPAS STO IDIO

        # forbid passing by afs if leaving depot, allow if returning to depot; forbid from afs to afs, not necessary but convenient
        mask[fs, 1:num_afs + 1] = 0#float('-inf')
        mask[afs, 0] = 1 #0

        mask[customer, :num_afs + 1] = 1 # phgaine ok # 0
        # edw ta kanei mhden enw emeis theloume 1!
        mask[fs] = torch.where(demands[fs] > 0, torch.zeros(mask[fs].size(), device=device), mask[fs])

        # path1: ->Node->Depot
        dis1 = distances[torch.arange(distances.size(0)), idx.squeeze(1)].clone()
        fuel_pd0 = cons_rate * dis1
        time_pd0 = dis1 / velocity
        dis1[:, num_afs + 1:] += distances[:, 0, num_afs + 1:]
        fuel_pd1 = cons_rate * dis1
        time_pd1 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + distances[:, 0, :]) / velocity
        time_pd1[:, 1:num_afs + 1] += 0.25
        time_pd1[:, num_afs + 1:] += 0.5

        # path2: ->Node-> Station-> Depot(choose the station making the total distance shortest)
        dis2 = distances[:, 1:num_afs + 1, :].gather(1, dis_by_afs[1].unsqueeze(1)).squeeze(1)
        dis2[:, 0] = 1 #0
        dis2 += distances[torch.arange(distances.size(0)), idx.squeeze(1)]
        fuel_pd2 = cons_rate * dis2
        time_pd2 = (distances[torch.arange(distances.size(0)), idx.squeeze(1)] + dis_by_afs[0]) / velocity
        time_pd2[:, 1:num_afs + 1] += 0.25
        time_pd2[:, num_afs + 1:] += 0.5

        # path3: ->Node-> Station-> Depot(choose the closest station to the node), ignore this path temporarily
        # the next node should be able to return to depot with at least one way; otherwise, mask it
        mask[~((fuel >= fuel_pd1) & (time >= time_pd1) | (fuel >= fuel_pd2) & (time >= time_pd2))] = 0#float('-inf')

        mask[(fuel < fuel_pd0) | (time < time_pd0)] = 0# float('-inf')

        #all_masked = mask[:, num_afs + 1:].eq(0).sum(1).le(0)
        all_masked = mask[:,num_afs + 1:].eq(0).all(dim=1)
        mask[all_masked, 0] = 1#0  # unmask the depot if all nodes are masked

        mask[mask == 0] = 1
        mask[mask == float('-inf')] = 0
        return dynamic

def cal_dis(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # degrees to radians
    lon1, lat1, lon2, lat2 = map(lambda x: x/180*np.pi, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = torch.pow(torch.sin(dlat / 2), 2) + torch.cos(lat1) * torch.cos(lat2) * torch.pow(torch.sin(dlon / 2), 2)
    c = 2 * 4182.44949 * torch.asin(torch.sqrt(a))    # miles
    # c = 2 * atan2(sqrt(a), sqrt(1 - a)) * 4182.44949
    return c

def reward_func2(tours, static, distances, beam_width=1):
    """
    :param tours: LongTensor, (batch*beam, seq_len)
    :param static: (batch, 2, num_nodes)
    :param distances: (batch, num_nodes, num_nodes)
    :param beam_width: set beam_width=1 when training
    :return: reward: Euclidean distance between each consecutive point， (batch)
    :return: locs: (batch, 2, seq_len)
    """
    bb_size, seq_len = tours.size()
    batch_size = static.size(0)
    depot = torch.zeros(bb_size, 1, dtype=torch.long, device=device)
    tours = torch.cat((depot, tours, depot), dim=1)         # start from depot, end at depot(although some have ended at depot)
    id0 = torch.arange(bb_size).unsqueeze(1).repeat(1, seq_len+1)
    reward = distances.repeat(beam_width, 1, 1)[id0, tours[:, :-1], tours[:, 1:]].sum(1)    # (batch*beam)
    # (batch*beam) -> (batch), choose the best reward
    reward, id_best = torch.cat(torch.chunk(reward.unsqueeze(1), beam_width, dim=0), dim=1).min(1)  # (batch)
    bb_idx = torch.arange(batch_size, device=device) + id_best * batch_size
    tours = tours[bb_idx]
    # print(tours)
    tours = tours.unsqueeze(1).repeat(1, static.size(1), 1)
    locs = torch.gather(static, dim=2, index=tours)  # (batch, 2, seq_len+)
    return reward, locs


def reward_func( static,tours, distances, beam_width=1):
    """
    :param tours: LongTensor, (batch*beam, seq_len)
    :param static: (batch, 2, num_nodes)
    :param distances: (batch, num_nodes, num_nodes)
    :param beam_width: set beam_width=1 when training
    :return: reward: Euclidean distance between each consecutive point， (batch)
    :return: locs: (batch, 2, seq_len)
    """
    bb_size, seq_len = tours.size()
    batch_size = static.size(0)
    depot = torch.zeros(bb_size, 1, dtype=torch.long, device=device)
    tours = torch.cat((depot, tours, depot), dim=1)         # start from depot, end at depot(although some have ended at depot)
    id0 = torch.arange(bb_size).unsqueeze(1).repeat(1, seq_len+1)
    reward = distances.repeat(beam_width, 1, 1)[id0, tours[:, :-1], tours[:, 1:]].sum(1)    # (batch*beam)

    # (batch*beam) -> (batch), choose the best reward
    reward, id_best = torch.cat(torch.chunk(reward.unsqueeze(1), beam_width, dim=0), dim=1).min(1)  # (batch)
    bb_idx = torch.arange(batch_size, device=device) + id_best * batch_size
    # print(tours)
    tours = tours[bb_idx]
    #print(tours)
    tours = tours.unsqueeze(1).repeat(1, static.size(1), 1)
    locs = torch.gather(static, dim=2, index=tours)  # (batch, 2, seq_len+)
    return reward#, locs


def render_func(locs, static, num_afs, save_dir="."):
    """
    :param locs: (batch, 2, sel_len)
    :param static: (batch, 2, num_nodes+1)
    :param num_afs: scalar
    :param save_dir: path to save figure
    :return: None
    """
    plt.close('all')
    data = locs[-1].cpu().numpy()   # (2, num_nodes+1), just plot the last one
    # demands = dynamic[-1, 1, 1:].cpu().numpy()*capacity   # (num_nodes), depot excluded
    # coords = static[-1][:, 1:].cpu().numpy().T     # (num_nodes, 2)
    plt.plot(data[0], data[1], zorder=1)
    origin_locs = static[-1, :, :].cpu().numpy()
    plt.scatter(origin_locs[0], origin_locs[1], s=4, c='r', zorder=2)
    plt.scatter(origin_locs[0, 0], origin_locs[1, 0], s=20, c='k', marker='*', zorder=3)  # depot
    plt.scatter(origin_locs[0, 1:num_afs+1], origin_locs[1, 1:num_afs+1], s=20, c='b', marker='+', zorder=4)    # afs
    for i, coords in enumerate(origin_locs.T[1:]):
        plt.annotate('%d' % (i+1), xy=coords, xytext=(2, 2), textcoords='offset points')  # mark numbers
    if os.path.isfile(save_dir):
        os.remove(save_dir)
    plt.axis('equal')
    plt.savefig(save_dir, dpi=400)


