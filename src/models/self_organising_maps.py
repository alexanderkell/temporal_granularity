from pathlib import Path
import logging

import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class SOMCalculator:
    """
    Interface for SOM Calculations.

    Interface to calculate SOM calculations

    """

    def __init__(self, data, n_clusters_dim_1, n_clusters_dim_2, batch_size):
        self.data = data
        self.n_clusters_dim_1 = n_clusters_dim_1
        self.n_clusters_dim_2 = n_clusters_dim_2
        self.batch_size = batch_size

    def train_som(self):
        """Get som object which is trained on data.

        Get som object which is trained on data. Set batch size for training as well as 
        number of clusters in both dimensions

        :param data: Data for training
        :type data: numpy ndarray
        :param n_clusters_dim_1: first dimension of clusters
        :type n_clusters_dim_1: int
        :param n_clusters_dim_2: second dimensino of clusters
        :type n_clusters_dim_2: int
        :param batch_size: batch size for training
        :type batch_size: int
        :return: Returns trained som object
        :rtype: MiniSom
        """
        som = MiniSom(self.n_clusters_dim_1, self.n_clusters_dim_2, 24, sigma=0.3,
                      learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)

        print(self.data)
        som.pca_weights_init(self.data)
        logger.info("Training SOM.")
        som.train_batch(self.data, self.batch_size,
                        verbose=True)  # random training
        logger.info("\n Trained SOM.")

        return som

    def get_representative_days(self, som, training_data, actions):

        win_map = som.win_map(training_data)
        print("actions: {}".format(actions))
        representative_days = []
        for position, action in zip(win_map.keys(), actions):
            print(position)
            median_day = np.median(win_map[position], axis=0)
            print(len(win_map[position]))

            print(cdist(median_day.reshape(
                1, -1), np.array(win_map[position]).reshape(len(win_map[position]), -1)))
            distance = cdist(median_day.reshape(
                1, -1), np.array(win_map[position]).reshape(len(win_map[position]), -1)).flatten()
            sorted_args = np.argsort(distance)
            sorted_dist = np.sort(distance)
            print(sorted_args)
            print(sorted_dist)
            k_smallest = np.argpartition(distance, action)[action]
            print(k_smallest)
            representative_days.append(k_smallest)
        return representative_days
