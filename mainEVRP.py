import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from Models.evrp.evrpActor import NetworkEVRP
from Tasks import evrp
from Tasks.evrp import EVRPDataset
from Models.critc import StateCritic
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

        # if render_fn is not None and batch_idx < num_plot:
        #     name = 'batch%d_%2.4f.png' % (batch_idx, reward)
        #     path = os.path.join(save_dir, name)
        #     render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)


def train(actor, critic, task, num_nodes, train_data, valid_data,
          reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm, experiment, weight_decay,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr,weight_decay=weight_decay)

    lr_decay = 1.0

    # lr_scheduler = CosineAnnealingLR(actor_optim,
    #                                   T_max=32,  # Maximum number of iterations.
    #                                   eta_min=1e-7)  # Minimum learning rate.
    #lr_scheduler = ExponentialLR(actor_optim, gamma=0.0099)# StepLR(actor_optim, step_size=20, gamma=5e-7) #optim.lr_scheduler.LambdaLR(actor_optim, lambda epoch: lr_decay ** epoch)
    # best so far:critic_lr_scheduler = ExponentialLR(critic_optim, gamma=0.99)#StepLR(critic_optim, step_size=20, gamma=0.01)

    #critic_lr_scheduler = ExponentialLR(critic_optim, gamma=0.099)

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
        start = epoch_start
        for batch_id, (static, dynamic, distances) in enumerate(train_loader):

            static = static.to(device)
            dynamic = dynamic.to(device)

            # Full forward pass through the dataset
            tour_indices, tour_logp, steps = actor(static, dynamic, distances)

            steps_per_epoch.append(steps)
            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static,tour_indices,distances) + steps

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

        #before_lr_critic = critic_optim.param_groups[0]["lr"]
        #before_lr_actor = actor_optim.param_groups[0]["lr"]

        # critic_lr_scheduler.step()
        # #lr_scheduler.step()
        # after_lr_critic = critic_optim.param_groups[0]["lr"]
        # #after_lr_actor = actor_optim.param_groups[0]["lr"]
        #
        # print("Epoch %d: CRITIC lr %f -> %f" % (epoch, before_lr_critic, after_lr_critic))
        # #print("Epoch %d: ACTOR lr %f -> %f" % (epoch, before_lr_actor, after_lr_actor))
        #
        # print(f'Learning rate for epoch {epoch}: critic {critic_lr_scheduler.get_lr()}')
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

    train_data = EVRPDataset(train_size, args.num_nodes,
                             t_limit, capacity,
                             num_afs, seed=args.seed)

    print('Train data: {}'.format(train_data))
    train_data = EVRPDataset(train_size, args.num_nodes, t_limit, capacity, num_afs)
    valid_data = EVRPDataset(args.valid_size, args.num_nodes, t_limit, capacity, num_afs, seed=args.seed+1)

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

    test_data = EVRPDataset(args.valid_size, args.num_nodes, t_limit, capacity, num_afs)


    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

    avg_tour_length = validate(test_loader, actor,
                          evrp.reward_func,
                          evrp.render_func,
                          test_dir, num_plot=5)

    print('Average tour length: ', avg_tour_length)
    # testing
    # finding variation
    # test_distances = []
    #
    # for i in range(10):
    #     test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    #
    #     avg_tour_length = test(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)
    #
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
    train_size = 40_000#0_000#000#000#15_000
    validation_size = 1000
    bss = [512] #[128, 256, 512]
    epochs = 30
    hidden_size = 128
    emb = ['conv']# , 'lin', 'graph']
    bs = 512
    critic_lr = 1e-1 # 0.05
    actor_lr = 5e-4
    # kati phge na ginei me auta
    # critic_lr = 5e-1
    # actor_lr = 5e-4

    # Average tour  length: 2307.4716796875, 128 bs

    # Average tour length: 2300.162841796875, 256bs
    #
    # me learning rate decay Average tour  length: 2391.3579711914062
    #2354.23828 with 1e-1 & 5e-6

    # Average tour length:  2266.9916381835938->  lr scheduler mono ston critic!
    # Average tour length  2353
    # all_both_lr 2282
    # kalutero mexri twra: Average tour length:  2118.3922729492188 -> ExponentialLR(critic_optim, gamma=0.099)

    # 1178 -> me 45.000

    #Average tour length:  1366.4947814941406-> me 45.000 kai critic lr decay
    weight_decays = [0, 0.05, 0.0001]
    num_nodes_arr = [20]#,50,100]
    # test = False
    # task="evrp"
    # parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    # parser.add_argument('--seed', default=12345, type=int)
    # parser.add_argument('--checkpoint', default=None)
    # parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--task', default='evrp')
    # parser.add_argument('--nodes', dest='num_nodes', default=num_nodes, type=int)
    # parser.add_argument('--actor_lr', default=actor_lr, type=float)
    # parser.add_argument('--critic_lr', default=critic_lr, type=float)
    # parser.add_argument('--max_grad_norm', default=2., type=float)
    # parser.add_argument('--batch_size', default=bs, type=int)  # 256
    # parser.add_argument('--hidden', dest='hidden_size', default=hidden_size, type=int)
    # parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    # parser.add_argument('--train-size', default=train_size, type=int)  # 0000
    # parser.add_argument('--valid-size', default=validation_size, type=int)
    # parser.add_argument('--embedding', default="conv")
    # embedding = "conv"
    # experiment_name = f'new_dataset_nodes{num_nodes}{embedding}size{train_size}_bs{bs}_actLr{actor_lr}_criLr{critic_lr}'
    # parser.add_argument('--experiment', default=experiment_name)
    #
    # args = parser.parse_args()
    #
    # train_vrp(args)
    for idx in range(len(weight_decays)):

        weight_decay = weight_decays[idx]
        num_nodes = num_nodes_arr[0]

        for i in range(len(emb)):
            embedding = emb[i]
            for j in range(len(bss)):
                bs = bss[j]
                parser = argparse.ArgumentParser(description='Combinatorial Optimization')
                parser.add_argument('--seed', default=12345, type=int)
                parser.add_argument('--checkpoint', default=None)
                parser.add_argument('--test', action='store_true', default=False)
                parser.add_argument('--task', default='evrp')
                parser.add_argument('--nodes', dest='num_nodes', default=num_nodes, type=int)
                parser.add_argument('--actor_lr', default=actor_lr, type=float)
                parser.add_argument('--critic_lr', default=critic_lr, type=float)
                parser.add_argument('--weight_decay', default=weight_decay, type=float)
                parser.add_argument('--max_grad_norm', default=2., type=float)
                parser.add_argument('--batch_size', default=bs, type=int)  # 256
                parser.add_argument('--hidden', dest='hidden_size', default=hidden_size, type=int)
                parser.add_argument('--dropout', default=0.1, type=float)
                parser.add_argument('--layers', dest='num_layers', default=1, type=int)
                parser.add_argument('--train-size', default=train_size, type=int)  # 0000
                parser.add_argument('--valid-size', default=validation_size, type=int)
                parser.add_argument('--embedding', default=embedding)

                experiment_name = f'weightdecayexp_EVRP_nodes{num_nodes}{embedding}size{train_size}_bs{bs}_actLr{actor_lr}_criLr{critic_lr}'
                parser.add_argument('--experiment', default=experiment_name)

                args = parser.parse_args()

                train_vrp(args)

    metric_labels = ["Distances", "Critic loss", "Critic estimate", "Loss", "Reward"]
    #labels = ['Conv embedding', 'Linear embedding', 'Graph embedding']
    # labels = ['10 nodes', '50 nodes', '100 nodes']
    labels = ['Without weight decay', 'With 0.05 weight decay', 'With 0.0001 weight decay']
    plot_models_metrics(metrics_per_model[0],
                        metrics_per_model[1],
                        metrics_per_model[2],
                        labels,
                        metric_labels,
                        experiment_name)

    k = len(metrics_per_model)

    for i in range(k):
        print(metrics_per_model[i])
