import pandas as pd
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import signal
from scipy import stats

from src.metrics.single_metrics import SingleMetrics
from itertools import combinations

logger = logging.getLogger(__name__)

class MultiMetrics:

    def __init__(self, solar, wind, load):
        self.solar = solar
        self.wind = wind
        self.load = load 
        
    def get_correlations(self):
        combination = list(combinations([self.solar, self.wind, self.load], r=2))
        
        correlations = []
        for comb in combination:
            correlations.append(self._calculate_correlation(comb[0], comb[1]))
        
        return correlations
    
    def _calculate_correlation(self, time_series_1, time_series_2):
        corr = stats.pearsonr(time_series_1.capacity_factor, time_series_2.capacity_factor)[0]
        return corr