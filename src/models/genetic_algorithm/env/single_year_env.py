"""
 Description: An environment to allow a machine learning algorithm to calculate best distance from cluster centre for a single year.

 Created on Fri Apr 05 2019

 Copyright (c) 2019 Newcastle University
 License is MIT
 Email is alexander@kell.es
"""

from src.metrics.metrics import MultiMetrics


class SingleYearEnv():
    """
    Environment to calculate error metrics from each step made.

    A single year environment which returns the error metrics based on representative days chosen. Extends base env.

    """

    def __init__():
        """
        Initialise the single year env.

        Instantiation of SingleYearEnv object which returns reward for each step. In this case, the reward being error metrics.

        """
        pass

    def step(action):
        MultiMetrics()
