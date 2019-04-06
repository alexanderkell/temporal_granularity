import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.spatial.distance import cdist
from src.models.self_organising_maps import SOMCalculator
from src.metrics.metrics import Metrics
from src.models.approximations import ApproximateData

import logging

logger = logging.getLogger(__name__)

"""
 Description: An environment to allow a machine learning algorithm to calculate best distance from cluster centre for a single year.

 Created on Fri Apr 05 2019

 Copyright (c) 2019 Newcastle University
 License is MIT
 Email is alexander@kell.es
"""


class SingleYearEnv():
    """
    Environment to calculate error metrics from each step made.

    A single year environment which returns the error metrics based on representative days chosen. Extends base env.

    """

    def __init__(self, solar, onshore, load, solar_df, onshore_df, load_df, n_clusters_dim_1, n_clusters_dim_2, batch_size):
        """
        Initialise the single year env.

        Instantiation of SingleYearEnv object which returns reward for each step. In this case, the reward being error metrics.

        """
        self.solar = solar
        self.onshore = onshore
        self.load = load

        self.np_data_list = [self.solar, self.onshore, self.load]

        self.solar_df = solar_df
        self.onshore_df = onshore_df
        self.load_df = load_df

        self.df_data_list = [self.solar_df, self.onshore_df, self.load_df]

        self.solar_som_calculator = SOMCalculator(
            self.solar, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.solar_som = self.solar_som_calculator.train_som()

        self.onshore_som_calculator = SOMCalculator(
            self.onshore, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.onshore_som = self.onshore_som_calculator.train_som()

        self.load_som_calculator = SOMCalculator(
            self.load, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.load_som = self.load_som_calculator.train_som()

        self.calculators = [self.solar_som_calculator,
                            self.onshore_som_calculator, self.load_som_calculator]
        self.som_objects = [self.solar_som, self.onshore_som, self.load_som]

    def step(self, actions):
        """Step through environment

        Selects representative days based upon optimisation input

        :param action: Distance from median to select representative day
        :type action: list float
        :return: Reward based on error metrics
        :rtype: list float
        """

        representative_data = []
        original_data = []

        actions = np.array(actions).reshape(3, -1)

        for np_data, df_data, calculator, som, action in zip(self.np_data_list, self.df_data_list, self.calculators, self.som_objects, actions):

            representative_days, cluster_numbers = calculator.get_representative_days(
                som, np_data, action)

            representative_days = pd.DataFrame(representative_days)

            representative_days = self.wide_to_long(representative_days)
            approximation_calc = ApproximateData(df_data, 4)
            representative_days = ApproximateData(df_data, 4).get_load_duration_curve(
                representative_days, cluster_numbers)

            representative_data.append(representative_days)

            original_days = approximation_calc.get_load_duration_curve(
                year="2014")

            original_data.append(original_days)

        metrics_calculator = Metrics(original_data[0], representative_data[0], original_data[1],
                                     representative_data[1], original_data[2], representative_data[2], "dc")

        error_metrics = metrics_calculator.get_mean_error_metrics()

        reward = -error_metrics.value.sum()
        logger.info("reward: {}".format(reward))

        return reward

    def wide_to_long(self, representative_load):
        representative_load = pd.melt(
            representative_load, value_vars=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], id_vars='cluster', value_name="capacity_factor")
        return representative_load
