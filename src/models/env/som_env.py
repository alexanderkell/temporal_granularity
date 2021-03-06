import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}'.format(project_dir))

project_dir = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minisom import MiniSom
from scipy.spatial.distance import cdist
from src.models.manipulations.self_organising_maps import SOMCalculator
from src.metrics.metrics import Metrics
from src.models.manipulations.approximations import ApproximateData
from src.models.env.env import Env
from src.models.run.long_term_optimisation.compare_methods import get_each_ldc
from src.metrics.multi_year_metrics import MultiYearMetrics


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


class SOMEnv(Env):
    """
    Environment to calculate error metrics from each step made.

    A single year environment which returns the error metrics based on representative days chosen. Extends base env.

    """

    def __init__(self, solar, onshore, load, solar_df, onshore_df, load_df, n_clusters_dim_1, n_clusters_dim_2, batch_size, year_start):
        """
        Initialise the single year env.

        Instantiation of SingleYearEnv object which returns reward for each step. In this case, the reward being error metrics.

        """
        super().__init__(solar, onshore, load, solar_df, onshore_df, load_df)

        # self.solar = solar
        # self.onshore = onshore
        # self.load = load

        # self.np_data_list = [self.solar, self.onshore, self.load]

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

        self.year_start = year_start

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

            # original_days = approximation_calc.get_load_duration_curve(
                # year="2013")



            # original_data.append(original_days)

        # metrics_calculator = Metrics(original_data[0], representative_data[0], original_data[1],
                                    #  representative_data[1], original_data[2], representative_data[2], "dc")

        pv_original = pd.read_csv(
            '{}data/processed/resources/pv_processed.csv'.format(project_dir))
        wind_original = pd.read_csv(
            '{}data/processed/resources/onshore_processed.csv'.format(project_dir))
        load_original = pd.read_csv(
            '{}data/processed/demand/load_NG/load_processed_normalised.csv'.format(project_dir))

        pv_original_ldcs, wind_original_ldcs, load_original_ldcs = get_each_ldc(pv_original, wind_original, load_original)

        multi_year_metrics_calculator = MultiYearMetrics(pv_original_ldcs, representative_data[0], wind_original_ldcs, representative_data[1], load_original_ldcs, representative_data[2], self.year_start)
        multi_year_metrics = multi_year_metrics_calculator.get_multi_year_average_metrics("dc")
        multi_year_metrics = multi_year_metrics.reset_index()
        # logger.debug("multi_year_metrics: \n{}".format(multi_year_metrics))

        nrmse = multi_year_metrics[multi_year_metrics['metric'] == 'nrmse dc'].iloc[0].value
        rae = multi_year_metrics[multi_year_metrics['metric'] == 'rae dc'].iloc[0].value
        correlation = multi_year_metrics[multi_year_metrics['metric'] == 'correlation'].iloc[0].value

        # error_metrics = metrics_calculator.get_mean_error_metrics()
        # nrmse = error_metrics.iloc[1].value
        # rae = error_metrics.iloc[2].value
        # correlation = error_metrics.iloc[0].value
        # reward = -error_metrics.value.sum()
        # logger.info("error_metrics: {}".format(error_metrics))
        # logger.info("error_metrics: {}".format(error_metrics.iloc[0]))

        # return reward
        return nrmse, rae, correlation

    def wide_to_long(self, representative_load):
        representative_load = pd.melt(
            representative_load, value_vars=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], id_vars='cluster', value_name="capacity_factor")
        return representative_load
