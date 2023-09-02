import matplotlib.pyplot as plt
import numpy as np
import torch

import math
import numpy as np
import datetime

from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix



def create_distance_matrix(points):
    return distance_matrix(points, points)

probs = torch.randint(10, (1,4)) #torch.ones(2,5)
import imageio
import os

def showGIF(image_directory):

    # Get all PNGs from the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]
    image_files.sort()  # Make sure they're in the order you want!

    images = []
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        images.append(imageio.imread(image_path))

    # Create a GIF
    output_file = 'output2.gif'
    kargs = {'duration': 1000}
    #imageio.mimsave(exportname, frames, 'GIF', **kargs)
    imageio.mimsave(output_file, images, 'GIF', **kargs)  # Duration is the amount of time each image shows (in seconds)
#showGIF("distributions")

def get_distance_of_each_city_from_current_city(static, current_position):
    ''' static: shape [2, num_nodes] '''
    distance_matrix = create_distance_matrix(static.transpose(1,0))

    return distance_matrix[:, current_position]

# IDEA: Se kathe polh na exeis kapoia xarakthristika
# px posh apostash exei apo ekei pou eimaste, ti load kai demand exoume

def explain_decision(step, static, dynamic, distribution, current_tour,chosen_city):

    tour = current_tour.detach().numpy()

    current_position = tour[-1]

    distance_of_each_city_from_current_city = get_distance_of_each_city_from_current_city(static, current_position)

    plot_categorical_distribution_with_descriptions_at_categories(step,
                                                                  distribution,
                                                                  dynamic,
                                                                  chosen_city,
                                                                  current_position,
                                                                  distance_of_each_city_from_current_city)


def plot_categorical_distribution_with_descriptions_at_categories(step, categorical_distribution,dynamic, chosen_city,current_position, distance_of_each_city_from_current_city):
    probs = categorical_distribution.probs

    probs= probs.detach().numpy()[0]
    categories = range(len(probs))
    fig, ax = plt.subplots()
    ax.bar(categories, probs)
    # ax.set_xlabel('Potential cities to visit')
    # ax.set_ylabel('Probability to visit each city')
    ax.set_xlabel('Πόλεις που μπορεί να επισκεφτεί')
    ax.set_ylabel('Πιθανότητα να επισκεφτεί κάθε πόλη')
    ax.set_xticks(categories)

    load = dynamic[:,0][0].item()
    ax.set_title(f"Φορτίο:{load:.2f}", fontsize=8, y=1.08)

    index_of_closest_city = np.argmin(distance_of_each_city_from_current_city)
    for i, prob in enumerate(probs):
        current_bar_dynamics = dynamic[:,i]

        demand = current_bar_dynamics[1].item()
        text = f"Απόσταση: {distance_of_each_city_from_current_city[i]:.2f}\n" #{prob:.2f}"

        if demand< 0:
            demand = 0
        text += f'Ζήτηση:{demand:.2f}\n'
        facecolor = "wheat"

        if index_of_closest_city == i & index_of_closest_city!=current_position:
            text +="\nΚοντινότερη"
            facecolor = "green"
        if chosen_city.item() == i:
            text += "\nΕπιλογή"
            facecolor = "red"

            # Calculate vertical offset for annotation.
        offset = 0.3 * (1 if i % 2 == 0 else 1.5)
        vertical_position =   offset

        props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
        ax.annotate(text, (i, vertical_position), ha='center', va='bottom', fontsize=7, bbox=props)

        # props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
        # ax.annotate(text, (i, prob), ha='center', va='bottom', fontsize=7, bbox=props)


    fig.tight_layout()
    plt.savefig(f"distribution_epoch{step}.png")

def plot_distribution(distribution, title='Distribution Plot'):
    probabilities = distribution.probs.detach().numpy()[0]
    categories = np.arange(len(probabilities))
    plt.bar(categories, probabilities)
    plt.xlabel('Categories')
    plt.ylabel('Probabilities')
    plt.title(title)


    plt.show()





# distrb = torch.distributions.Categorical(probs)
# plot_distribution(distrb)

