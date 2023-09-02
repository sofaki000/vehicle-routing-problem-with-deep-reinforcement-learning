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
from main import test
from utils.plots import plot_metrics

batch_size = 25
valid_size = 1000
num_nodes = 10
LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
MAX_DEMAND = 9
STATIC_SIZE = 2  # (x, y)
DYNAMIC_SIZE = 2  # (load, demand)

max_load = LOAD_DICT[num_nodes]

test_distances = []

if __name__ == '__main__':

    test_data = VehicleRoutingDataset(valid_size,
                                      num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      12345 + 2)
    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    128,
                    test_data.update_dynamic,
                    test_data.update_mask,
                    'conv',
                    num_layers=1)
    actor.load_state_dict(torch.load("actorconvsize15000_bs256_actLr0.0005_criLr0.05.pt"))
    actor.eval()

    for i in range(10):

        test_loader = DataLoader(test_data,
                                 batch_size,
                                 False,
                                 num_workers=0)

        avg_tour_length = test(test_loader,
                               actor, vrp.reward,
                               vrp.render, "test", num_plot=5)

        print('Average tour length: ', avg_tour_length)

        test_distances.append(avg_tour_length)

        test_data = VehicleRoutingDataset(valid_size,
                                          num_nodes,
                                          max_load,
                                          MAX_DEMAND,
                                          12345 + 2+i*10)

    variance = np.var(test_distances)
    plt.clf()
    plt.tight_layout()
    plt.plot(test_distances)
    plt.title(f"Test distances, variance: {variance:.3f}")
    plt.savefig("testDistances.png")

