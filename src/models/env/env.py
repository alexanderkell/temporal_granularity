import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd

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


class Env:
    """
    Environment to calculate error metrics from each step made.

    A single year environment which returns the error metrics based on representative days chosen. Extends base env.

    """

    def __init__(self, solar, onshore, load, solar_df, onshore_df, load_df):
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

    def step(self):
        """Step through environment

        Selects representative days based upon optimisation input

        :param action: Distance from median to select representative day
        :type action: list float
        :return: Reward based on error metrics
        :rtype: list float
        """
        pass

    def wide_to_long(self, representative_load):
        representative_load = pd.melt(
            representative_load, value_vars=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], id_vars='cluster', value_name="capacity_factor")
        return representative_load
