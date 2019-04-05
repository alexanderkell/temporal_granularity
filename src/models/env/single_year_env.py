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

    def __init__(self, solar, wind, load, n_clusters_dim_1, n_clusters_dim_2):
        """
        Initialise the single year env.

        Instantiation of SingleYearEnv object which returns reward for each step. In this case, the reward being error metrics.

        """
        self.solar = solar
        self.wind = wind
        self.load = load

        self.som_solar = SOMCalculator()

        self.som = self.som_calculator.train_som()

    def step(self, action):
        representative_solar = self.som_calulator.get_representative_days(
            self.som, self.solar, action)
        representative_wind = self.som_calulator.get_representative_days(
            self.som, self.wind, action)
        representative_load = self.som_calulator.get_representative_days(
            self.som, self.load, action)

        metrics_calculator = Metrics(self.solar, representative_solar, self.wind,
                                     representative_wind, self.load, representative_load, "dc")

        error_metrics = metrics_calculator.get_mean_error_metrics()
        logger.info("error_metrics: {}".format(error_metrics))
        return 1
