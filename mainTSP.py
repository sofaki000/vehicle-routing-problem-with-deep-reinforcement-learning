import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Models.actor import DRL4TSP
from Models.tsp.tspActor import NetworkTSP
from Models.tsp.tspPointerNetwork import CombinatorialRLTSP
from Tasks import vrp, tsp
from Models.critc import StateCritic, StateCriticTSP
from Tasks.tsp import TSPDataset
from utils.plots import plot_metrics, plot_models_metrics
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))

metrics_per_model = []


def test(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, static in enumerate(data_loader):
        static = static.to(device)
        with torch.no_grad():
            tour_indices, _ = actor.forward(static, static)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, reward)
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

        static = batch

        static = static.to(device)

        with torch.no_grad():
            reward, probs, actions, actions_idxs = actor.forward(static)

        #reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        # if render_fn is not None and batch_idx < num_plot:
        #     name = 'batch%d_%2.4f.png' % (batch_idx, reward)
        #     path = os.path.join(save_dir, name)
        #     render_fn(static, actions_idxs, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
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

    lr_decay = 1.0
    # lr_scheduler = optim.lr_scheduler.LambdaLR(actor_optim, lambda epoch: lr_decay ** epoch)

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

        times, losses, rewards, critic_rewards, distances, critic_losses = [], [], [], [], [], []

        epoch_start = time.time()
        beta = 0.9

        for batch_idx, batch in enumerate(train_loader):

            static = Variable(batch)

            static = static.to(device)

            # Full forward pass through the dataset
            reward, probs, actions, actions_idxs =actor(static)

            if batch_idx == 0:
                critic_exp_mvg_avg = reward.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * reward.mean())

            # Sum the log probabilities for each city in the tour
            #reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            #critic_est = critic(static).view(-1)
            logprobs = 0
            for prob in probs:
                logprob = torch.log(prob)
                logprobs += logprob
            logprobs[logprobs < -1000] = 0.

            advantage = (reward - critic_exp_mvg_avg)#- critic_est)

            reinforce = advantage  * logprobs
            actor_loss = reinforce.mean() # torch.mean(advantage.detach() * tour_logp)

            #critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            #torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
            # critic_optim.zero_grad()
            # critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            # critic_optim.step()
            #
            # critic_rewards.append(torch.mean(critic_est.detach()).item())
            # critic_losses.append(torch.mean(critic_loss.detach()).item())

            # actor reward is the absolute log probabilities of the chosen actions multiplied
            # by the advantage
            actor_reward = torch.mean(torch.abs(reinforce))

            rewards.append(actor_reward.item())

            losses.append(torch.mean(actor_loss.detach()).item())
            distances.append(torch.mean(reward).item())

        # epoch finished
        # lr_scheduler.step()

        # print(f'Learning rate for epoch {epoch}={lr_scheduler.get_lr()}')
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

        print(
            f'Mean epoch {epoch}/{epochs}: loss={mean_loss}, reward={mean_reward}, distance={mean_distance}, val reward={mean_valid}')

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

    print('Starting TSP training')

    STATIC_SIZE = 2  # (x, y)
    train_data = TSPDataset(20, train_size,  args.seed + 1)
    valid_data = TSPDataset(20, args.valid_size,  args.seed + 2)

    # actor = NetworkTSP(STATIC_SIZE,
    #                 args.hidden_size,
    #                 None,
    #                 train_data.update_mask,
    #                 args.embedding,
    #                 args.num_layers,
    #                 args.dropout).to(device)

    embedding_size = 128
    hidden_size = 128
    n_glimpses = 1
    tanh_exploration = 10
    use_tanh = True
    batch_size = 128
    beta = 0.9
    max_grad_norm = 2.

    """Δημιουργούμε τα μοντέλα που θα εκπαιδεύσουμε."""

    actor = CombinatorialRLTSP(
        embedding_size,
        hidden_size,
        20,
        n_glimpses,
        tanh_exploration,
        use_tanh,
        tsp.reward,
        embedding="Graph",
        attention="Dot")

    # actor = CombinatorialRLTSP(
    #     args.hidden_size,
    #     args.hidden_size,
    #     20,
    #     1,
    #     10,
    #     True,
    #     tsp.reward,
    #     attention="Dot")

    print('Actor: {} '.format(actor))

    critic = StateCriticTSP(STATIC_SIZE, args.hidden_size, args.embedding).to(device)

    print('Critic: {}'.format(critic))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if not args.test:
        train(actor, critic, **kwargs)

    # test_data = TSPDataset( args.train_size, num_nodes, args.seed + 2)
    #
    # test_dir = 'test'

    # testing
    # finding variation
    test_distances = []

    # for i in range(10):
    #     test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    #     avg_tour_length = test(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)
    #     print('Average tour length: ', avg_tour_length)
    #
    #     test_distances.append(avg_tour_length)
    #
    #     variance = np.var(test_distances)
    #     plt.clf()
    #     plt.plot(test_distances)
    #     plt.title(f"Test distances, variance: {variance}")
    #     plt.savefig("testDistances.png")


if __name__ == '__main__':
    train_size = 15_000#_000
    num_nodess = [20]#, 50, 100]

    bss = [100]
    # bss = [256] -> oles kales oi metrikes ektos distances # [128, 256, 512]
    epochs = 20#0
    emb = ['conv']  # , 'lin', 'graph']
    critic_lr = 5e-1
    actor_lr = 1e-4

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
        parser.add_argument('--critic_lr', default=critic_lr, type=float)
        parser.add_argument('--max_grad_norm', default=2., type=float)
        parser.add_argument('--batch_size', default=bs, type=int)  # 256
        parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
        parser.add_argument('--dropout', default=0.1, type=float)
        parser.add_argument('--layers', dest='num_layers', default=1, type=int)
        parser.add_argument('--train-size', default=train_size, type=int)  # 0000
        parser.add_argument('--valid-size', default=1000, type=int)
        parser.add_argument('--embedding', default=embedding)

        experiment_name = f'TAHN_EXP_SP_CRITIC_AVG_nodes_{num_nodes}_{embedding}size{train_size}_bs{bs}_actLr{actor_lr}_criLr{critic_lr}'
        parser.add_argument('--experiment', default=experiment_name)

        args = parser.parse_args()

        train_vrp(args)

    # metric_labels = ["Distances", "Critic loss", "Critic reward", "Loss", "Reward"]
    # # labels = ['Conv embedding', 'Linear embedding', 'Graph embedding']
    # labels = ['10 nodes', '50 nodes', '100 nodes']
    #
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
