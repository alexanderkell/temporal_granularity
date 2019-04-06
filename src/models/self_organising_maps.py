from pathlib import Path
import logging
import functools
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

    @functools.lru_cache(maxsize=128, typed=False)
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

        # som.pca_weights_init(self.data)
        som.random_weights_init(self.data)
        som.train_batch(self.data, self.batch_size,
                        verbose=False)  # random training

        return som

    def get_representative_days(self, som, training_data, actions):

        win_map = som.win_map(training_data)
        representative_days = []
        cluster_numbers = []
        for i, (position, action) in enumerate(zip(win_map.keys(), actions)):
            median_day = np.median(win_map[position], axis=0)
            cluster_size = len(win_map[position])
            distance = cdist(median_day.reshape(
                1, -1), np.array(win_map[position]).reshape(len(win_map[position]), -1)).flatten()
            try:
                k_smallest = np.argpartition(distance, action)[action]
            except ValueError:
                k_smallest = np.argpartition(distance, 0)[-1]
            logger.debug("action: {}".format(action))
            # representative_days.append(k_smallest)
            logger.debug("k smallest day:".format(
                win_map[position][k_smallest]))
            representative_days.append(win_map[position][k_smallest])
            cluster_numbers.append(
                {"cluster_num": cluster_size, "cluster": i})

        representative_days = pd.DataFrame(
            representative_days).reset_index().rename(columns={"index": "cluster"})

        cluster_df = pd.DataFrame(cluster_numbers)

        cluster_df = cluster_df.reindex(
            cluster_df.index.repeat(cluster_df['cluster_num']))

        return representative_days, cluster_df
