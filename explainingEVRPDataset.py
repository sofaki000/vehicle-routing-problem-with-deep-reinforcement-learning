import os
import time
import argparse
import datetime
import torch
import torch.optim as optim

from Models.evrp.evrpActor import NetworkEVRP
from Tasks import vrp, evrp
from Tasks.vrp import VehicleRoutingDataset
from Models.critc import StateCritic
from utils.plots import plot_metrics, plot_models_metrics
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))

metrics_per_model = []
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EVRPDataset2(Dataset):
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
        # self.dynamic[:, 1, :self.num_afs+1] = 0

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




def test(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_id, (static, dynamic, distances) in enumerate(data_loader):

        static = static.to(device)
        dynamic = dynamic.to(device)

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, distances)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None:
            name = f'batch{reward:.4f}.png'
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, distances = batch

        static = static.to(device)
        dynamic = dynamic.to(device)

        with torch.no_grad():
            tour_indices, _ ,_= actor.forward(static, dynamic, distances)

        reward = reward_fn(static, tour_indices,distances).mean().item()
        rewards.append(reward)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data,
          reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, experiment,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)


    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_reward = np.inf

    losses_per_epoch = []
    rewards_per_epoch = []
    critic_losses_per_epoch = []
    critic_rewards_per_epoch = []
    distances_per_epoch = []


    for epoch in range(epochs):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards, distances_epoch, critic_losses, steps_per_epoch = [], [], [], [], [], [], []

        epoch_start = time.time()
        for batch_id, (static, dynamic, distances) in enumerate(train_loader):

            static = static.to(device)
            dynamic = dynamic.to(device)

            # Full forward pass through the dataset
            tour_indices, tour_logp, steps = actor(static, dynamic, distances)

            steps_per_epoch.append(steps)
            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static,tour_indices,distances)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)

            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))

            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            critic_losses.append(torch.mean(critic_loss.detach()).item())

            # actor reward is the absolute log probabilities of the chosen actions multiplied
            # by the advantage
            actor_reward = torch.mean(advantage.detach() * torch.abs(tour_logp.detach().sum(dim=1)))

            rewards.append(actor_reward.item())

            losses.append(torch.mean(actor_loss.detach()).item())
            distances_epoch.append(torch.mean(reward).item())



        # epoch finished


        mean_critic_reward = np.mean(critic_rewards)
        mean_critic_loss = np.mean(critic_losses)
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_distance = np.mean(distances_epoch)

        losses_per_epoch.append(mean_loss)
        rewards_per_epoch.append(mean_reward)
        distances_per_epoch.append(mean_distance)

        critic_losses_per_epoch.append(mean_critic_loss)
        critic_rewards_per_epoch.append(mean_critic_reward)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print(f'Mean epoch {epoch}/{epochs}: loss={mean_loss}, reward={mean_reward}, distance={mean_distance}, val reward={mean_valid}')

    plt.clf()

    metrics = [distances_per_epoch,
               critic_losses_per_epoch,
               critic_rewards_per_epoch,
               losses_per_epoch,
               rewards_per_epoch]

    labels = ["Distances", "Critic loss", "Critic estimate", "Actor loss", "Actor reward"]

    metrics_per_model.append(metrics)
    plot_metrics(metrics,
                 labels,
                 experiment,
                 labels,
                 title="CVRP",
                 xlabel="Epochs")
    plt.clf()
    plt.plot(losses_per_epoch)
    plt.savefig(f"losses{experiment}.png")
    plt.clf()

    plt.plot(steps_per_epoch)
    plt.savefig(f"steps_per_epoch{experiment}.png")
    plt.clf()

    plt.plot(rewards_per_epoch)
    plt.savefig(f"rewards{experiment}.png")
    plt.clf()

    plt.plot(distances_per_epoch)
    plt.savefig(f"distances{experiment}.png")
    plt.clf()

    plt.plot(critic_losses_per_epoch)
    plt.savefig(f"critic_losses{experiment}.png")
    plt.clf()

    plt.plot(critic_rewards_per_epoch)
    plt.savefig(f"critic_rewards{experiment}.png")
    plt.clf()

    torch.save(actor.state_dict(), f'actor{experiment_name}.pt')
    torch.save(critic.state_dict(), f'critic{experiment_name}.pt')


def train_vrp(args):
    print('Starting EVRP training')
    # Determines the maximum amount of load for a vehicle based on num nodes
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 3
    t_limit = 11
    capacity = 60
    num_afs = 3

    train_data = EVRPDataset2(train_size, args.num_nodes,
                             t_limit, capacity,
                             num_afs, seed=args.seed)

    print('Train data: {}'.format(train_data))
    train_data = EVRPDataset2(train_size, args.num_nodes, t_limit, capacity, num_afs)
    valid_data = EVRPDataset2(args.valid_size, args.num_nodes, t_limit, capacity, num_afs, seed=args.seed + 1)

    actor = NetworkEVRP(STATIC_SIZE,
                        DYNAMIC_SIZE,
                        args.hidden_size,
                        train_data.update_dynamic,
                        train_data.update_mask,
                        args.embedding,
                        args.num_layers,
                        args.dropout).to(device)
    print('Actor: {} '.format(actor))

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size, args.embedding).to(device)

    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = evrp.reward_func
    kwargs['render_fn'] = evrp.render_func

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = EVRPDataset2(args.valid_size, args.num_nodes, t_limit, capacity, num_afs)


    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

    avg_tour_length = validate(test_loader, actor,
                          evrp.reward_func,
                          evrp.render_func,
                          test_dir, num_plot=5)

    print('Average tour length: ', avg_tour_length)



if __name__ == '__main__':
    train_size = 10_000  # number of samples
    num_nodes = 10  # number of customers we have to visit
    t_limit = 11  # available time we have. Must visit depot to restart timer
    capacity = 60  # available
    num_afs = 2  # number of
    seed = 12345
    validation_size = 100

    bs = 256
    epochs = 20
    hidden_size = 128
    emb = ['conv']
    critic_lr = 1e-1
    actor_lr = 5e-5
    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {3: 10,5: 10, 10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 3

    max_load = LOAD_DICT[num_nodes]

    test = False
    task="evrp"
    nodes =num_nodes
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='evrp')
    parser.add_argument('--nodes', dest='num_nodes', default=num_nodes, type=int)
    parser.add_argument('--actor_lr', default=actor_lr, type=float)
    parser.add_argument('--critic_lr', default=critic_lr, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=bs, type=int)  # 256
    parser.add_argument('--hidden', dest='hidden_size', default=hidden_size, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=train_size, type=int)  # 0000
    parser.add_argument('--valid-size', default=validation_size, type=int)
    parser.add_argument('--embedding', default="conv")
    embedding = "conv"
    experiment_name = f'new_dataset_nodes{num_nodes}{embedding}size{train_size}_bs{bs}_actLr{actor_lr}_criLr{critic_lr}'
    parser.add_argument('--experiment', default=experiment_name)

    args = parser.parse_args()

    train_vrp(args)

