import pandas as pd
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, auc
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
        
        self.original = self.original.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
        self.representative = self.representative.replace([np.inf, -np.inf], np.nan).fillna(method="ffill")

        nrmse = sqrt(mean_squared_error(self.original['capacity_factor'], self.representative['capacity_factor']))/(self.original['capacity_factor'].max()-self.original['capacity_factor'].min())
        nrmse = nrmse*100
        return nrmse

    def rae(self):
        # rae = abs((self.original['capacity_factor'].sum()-self.representative['capacity_factor'].sum())/self.original['capacity_factor'].sum())
        # logger.info("original_head: \n {}".format(self.original.head()))
        rae = abs((auc(self.original.index_for_year, self.original['capacity_factor'])-auc(self.representative.index_for_year, self.representative['capacity_factor'])))/auc(self.original.index_for_year, self.original['capacity_factor'])
        # rae = auc(self.original.datetime ,self.original['capacity_factor'])

        # rae = abs((self.original.capacity_factor.mean())-(self.representative.capacity_factor.mean())/(self.original.capacity_factor.mean()))

        rae = rae*100
        return rae

    # def _approximal(self, ldc):
    #     largest_entry = ldc.iloc[-1]
    #     smallest_entry = ldc.iloc[0]
    #     logger.debug("largest_entry: \n{}".format(largest_entry))
    #     logger.debug("smallest_entry: \n{}".format(smallest_entry))

    #     bin_size = (largest_entry-smallest_entry)/self.num_bins

    #     approximatal = np.cumsum([bin_size for _ in range(self.num_bins)])
        
        
    #     logger.debug("test: \n{}".format(approximatal))
    #     return 1
    
