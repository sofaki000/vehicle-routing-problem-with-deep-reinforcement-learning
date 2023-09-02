import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from Models.actor import DRL4TSP
from Tasks import vrp
from Tasks.vrp import VehicleRoutingDataset
from Models.critc import StateCritic
from utils.plots import plot_metrics, plot_models_metrics

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))



metrics_per_model = []
def test(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    logp = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, tour_logp = actor.forward(static, dynamic, x0)

        logp.append(tour_logp)
        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
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

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        # if render_fn is not None and batch_idx < num_plot:
        #     name = 'batch%d_%2.4f.png'%(batch_idx, reward)
        #     path = os.path.join(save_dir, name)
        #     render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,experiment,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)#, weight_decay=1e-3) #new: added weight decay
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)#, weight_decay=1e-3)


    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)



    best_reward = np.inf

    losses_per_epoch = []
    rewards_per_epoch = []
    critic_losses_per_epoch = []
    critic_rewards_per_epoch =  []
    distances_per_epoch = []


    for epoch in range(epochs):

            actor.train()
            critic.train()

            times, losses, rewards, critic_rewards, distances, critic_losses =[], [], [], [], [], []

            epoch_start = time.time()
            start = epoch_start

            for batch_idx, batch in enumerate(train_loader):

                static, dynamic, x0 = batch

                static = static.to(device)
                dynamic = dynamic.to(device)
                x0 = x0.to(device) if len(x0) > 0 else None

                # Full forward pass through the dataset
                tour_indices, tour_logp = actor(static, dynamic, x0)

                # Sum the log probabilities for each city in the tour
                reward = reward_fn(static, tour_indices)

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
                distances.append(torch.mean(reward).item())

            # epoch finished
            #lr_scheduler.step()

            #print(f'Learning rate for epoch {epoch}={lr_scheduler.get_lr()}')
            mean_critic_reward = np.mean(critic_rewards)
            mean_critic_loss = np.mean(critic_losses)
            mean_loss = np.mean(losses)
            mean_reward = np.mean(rewards)
            mean_distance = np.mean(distances)

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

    labels = ["Distances","Critic loss", "Critic estimate", "Actor loss", "Actor reward"]

    metrics_per_model.append(metrics)
    plot_metrics(metrics,
                 labels,
                 experiment,
                 labels,
                 title="CVRP",
                 xlabel="Epochs")
    plt.clf()
    plt.grid()
    plt.plot(losses_per_epoch)
    plt.savefig(f"losses{experiment}.png")
    plt.clf()

    plt.grid()
    plt.plot(rewards_per_epoch)
    plt.savefig(f"rewards{experiment}.png")
    plt.clf()

    plt.grid()
    plt.plot(distances_per_epoch)
    plt.savefig(f"distances{experiment}.png")
    plt.clf()

    plt.grid()
    plt.plot(critic_losses_per_epoch)
    plt.savefig(f"critic_losses{experiment}.png")
    plt.clf()

    plt.grid()
    plt.plot(critic_rewards_per_epoch)
    plt.savefig(f"critic_rewards{experiment}.png")
    plt.clf()

    torch.save(actor.state_dict(), f'actor{experiment_name}.pt')
    torch.save(critic.state_dict(), f'critic{experiment_name}.pt')


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    print('Starting VRP training')

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = DRL4TSP(STATIC_SIZE,
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
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)



    testAvgTourLengthOfModel(actor,args.num_nodes )

def testAvgTourLengthOfModel(actor,num_nodes, plot=True, filename=None):
    test_dir = 'testCVRP'
    # testing
    # finding variation
    test_distances = []
    batch_size = 1
    MAX_DEMAND = 9
    seed = 10
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    num_nodes_100 = 100
    max_load = LOAD_DICT[num_nodes_100]
    test_data_10 = VehicleRoutingDataset(10,
                                         num_nodes,
                                         max_load,
                                         MAX_DEMAND,
                                         seed + 1)

    test_loader = DataLoader(test_data_10, batch_size, False, num_workers=0)

    print(f'For {num_nodes} nodes:')
    for i in range(20):
        avg_tour_length = test(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)
        print('Average tour length: ', avg_tour_length)

        test_distances.append(avg_tour_length)

    variance = np.var(test_distances)

    print(f'variance of model {filename} tested for {num_nodes} nodes is {variance}')
    print(f'mean distance is {np.mean(test_distances)}')

    if plot:
        plt.clf()
        plt.plot(test_distances)
        plt.title(f"Test distances, variance: {variance}")
        plt.savefig(f"testDistances{filename}.png")


if __name__ == '__main__':

    train_size = 80_000
    num_nodess = [50]#, 50, 100]

    bss = [256] #[128, 256, 512]
    epochs = 15
    emb = ['graph'] #, 'lin', 'conv']
    critic_lr = 5e-1#2
    actor_lr = 5e-5#4

    for num_nodes_idx in range(len(num_nodess)):
        num_nodes = num_nodess[num_nodes_idx]
        bs = bss[0]
        embedding = emb[0]
        parser = argparse.ArgumentParser(description='Combinatorial Optimization')
        parser.add_argument('--seed', default=12345, type=int)
        parser.add_argument('--checkpoint', default=None)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--task', default='vrp')
        parser.add_argument('--nodes', dest='num_nodes', default=num_nodes, type=int)
        parser.add_argument('--actor_lr', default=actor_lr, type=float)
        parser.add_argument('--critic_lr', default=critic_lr ,type=float)
        parser.add_argument('--max_grad_norm', default=2., type=float)
        parser.add_argument('--batch_size', default=bs, type=int)# 256
        parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--layers', dest='num_layers', default=1, type=int)
        parser.add_argument('--train-size',default=train_size, type=int)#0000
        parser.add_argument('--valid-size', default=1000, type=int)
        parser.add_argument('--embedding', default=embedding)

        experiment_name = f'CVRP_node{num_nodes}_{train_size}_bs{bs}_actLr{actor_lr}_criLr{critic_lr}'
        parser.add_argument('--experiment', default=experiment_name )

        args = parser.parse_args()
    
        train_vrp(args)

    metric_labels = ["Distances", "Critic loss", "Critic estimate", "Loss", "Reward"]
    # labels = ['Conv embedding', 'Linear embedding', 'Graph embedding']
    labels = ['10 nodes', '50 nodes', '100 nodes']

    # plot_models_metrics(metrics_per_model[0],
    #                     metrics_per_model[1],
    #                     metrics_per_model[2],
    #                     labels,
    #                     metric_labels,
    #                     experiment_name)
    #
    # k = len(metrics_per_model)
    #
    # for i in range(k):
    #     print(metrics_per_model[i])
