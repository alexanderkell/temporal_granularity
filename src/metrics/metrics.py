import pandas as pd
import logging
import numpy as np
from scipy import stats

from src.metrics.single_metrics import SingleMetrics
from src.metrics.multi_metrics import MultiMetrics

from itertools import combinations

logger = logging.getLogger(__name__)

class Metrics:
    
    def __init__(self, original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load):
        self.original_solar = original_solar
        self.representative_solar = representative_solar
        self.original_wind = original_wind
        self.representative_wind = representative_wind
        self.original_load = original_load
        self.representative_load = representative_load

        self.results = pd.DataFrame(columns = ['series_type', 'metric', 'value'])

    def get_error_metrics(self):
        pass

    def get_nrmse(self):
        nrmse_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            nrmse_result={}
            solar_metrics = SingleMetrics(original[1], representative[1]).nrmse()
            nrmse_result.update({'series_type': data_type,'metric':'nrmse','value':solar_metrics})
            nrmse_results.append(nrmse_result)

        return nrmse_results   
    
    def get_rae(self):
        rae_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            rae_result={}
            solar_metrics = SingleMetrics(original[1], representative[1]).rae()
            rae_result.update({'series_type': data_type,'metric':'rae','value':solar_metrics})
            logger.debug(rae_result)
            rae_results.append(rae_result)
        
        return rae_results