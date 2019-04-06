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

        self.solar_df = solar_df
        self.onshore_df = onshore_df
        self.load_df = load_df

        self.solar_som_calculator = SOMCalculator(
            self.solar, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.solar_som = self.solar_som_calculator.train_som()

        self.onshore_som_calculator = SOMCalculator(
            self.onshore, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.onshore_som = self.onshore_som_calculator.train_som()

        self.load_som_calculator = SOMCalculator(
            self.load, n_clusters_dim_1, n_clusters_dim_2, batch_size)
        self.load_som = self.load_som_calculator.train_som()

    def step(self, action):
        """Step through environment

        Selects representative days based upon optimisation input

        :param action: Distance from median to select representative day
        :type action: list float
        :return: Reward based on error metrics
        :rtype: list float
        """

        representative_solar = pd.DataFrame(self.solar_som_calculator.get_representative_days(
            self.solar_som, self.solar, action[0]))
        representative_wind = pd.DataFrame(self.onshore_som_calculator.get_representative_days(
            self.onshore_som, self.onshore, action[1]))
        representative_load = pd.DataFrame(self.load_som_calculator.get_representative_days(
            self.load_som, self.load, action[2]))

        logger.info("load: {}".format(self.load_df.head()))

        representative_solar = self.wide_to_long(representative_solar)

        representative_wind = self.wide_to_long(representative_wind)

        representative_load = self.wide_to_long(representative_load)

        # logger.info("solar_df: \n{}".format(self.solar_df.head()))

        # self.solar_df = self.wide_to_long(self.solar_df)
        # self.onshore_df = self.wide_to_long(self.onshore_df)
        # self.load_df = self.wide_to_long(self.load_df)

        logger.info("representative_load: {}".format(representative_load))

        metrics_calculator = Metrics(self.solar_df, representative_solar, self.onshore_df,
                                     representative_wind, self.load_df, representative_load, "dc")

        error_metrics = metrics_calculator.get_mean_error_metrics()
        logger.info("error_metrics: {}".format(error_metrics))
        return 1

    def wide_to_long(self, representative_load):
        representative_load = pd.melt(
            representative_load, value_vars=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], value_name="capacity_factor")
        return representative_load
