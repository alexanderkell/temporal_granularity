import pandas as pd
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import signal
from scipy import stats

logger = logging.getLogger(__name__)

class SingleMetrics:

    def __init__(self, original, representative):
        self.original = original
        self.representative = representative
        # self.binned_original = self._approximaal(self.original)

    def nrmse(self):
        nrmse = sqrt(mean_squared_error(self.original['capacity_factor'], self.representative['capacity_factor']))/(self.original['capacity_factor'].max()-self.original['capacity_factor'].min())
        return nrmse

    def rae(self):
        rae = abs((self.original['capacity_factor'].sum()-self.representative['capacity_factor'].sum())/self.original['capacity_factor'].sum())
        return rae

    def display_metrics(self):
        logger.info("")

    # def _approximaal(self, ldc):
    #     largest_entry = ldc.iloc[-1]
    #     smallest_entry = ldc.iloc[0]
    #     logger.debug("largest_entry: \n{}".format(largest_entry))
    #     logger.debug("smallest_entry: \n{}".format(smallest_entry))

    #     bin_size = (largest_entry-smallest_entry)/self.num_bins

    #     approximatal = np.cumsum([bin_size for _ in range(self.num_bins)])
        
        
    #     logger.debug("test: \n{}".format(approximatal))
    #     return 1
    
