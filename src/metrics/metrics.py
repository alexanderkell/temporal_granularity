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

        

    def get_error_metrics(self):
        
        # results = pd.DataFrame(columns = ['series_type', 'metric', 'value'])
        metrics = []

        all_nrmse = self._get_nrmse()
        all_rae = self._get_rae()
        all_corr = self._calculate_correlation_error()

        metrics.extend(all_nrmse)
        metrics.extend(all_rae)
        metrics.extend(all_corr)

        results = pd.DataFrame(metrics)
        return results


    def _get_nrmse(self):
        nrmse_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            nrmse_result={}
            solar_metrics = SingleMetrics(original[1], representative[1]).nrmse()
            nrmse_result.update({'series_type': data_type,'metric':'nrmse','value':solar_metrics})
            nrmse_results.append(nrmse_result)

        return nrmse_results   
    
    def _get_rae(self):
        rae_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            rae_result={}
            rae_value = SingleMetrics(original[1], representative[1]).rae()
            rae_result.update({'series_type': data_type,'metric':'rae','value': rae_value})
            rae_results.append(rae_result)
        
        return rae_results

    def _get_correlations(self, data_type):
        
        all_correlations = []
        if data_type == "original":
            original_correlations = MultiMetrics(self.original_solar, self.original_wind, self.original_load).get_correlations()
            all_correlations.extend(original_correlations)
        elif data_type == "representative":
            representative_correlations = MultiMetrics(self.representative_solar, self.representative_wind, self.representative_load).get_correlations()
            all_correlations.extend(representative_correlations)
        else:
            raise ValueError("data_type can not be {}. It must be original or representative".format(data_type))
        return all_correlations

    def _calculate_correlation_error(self):
        original_correlations = self._get_correlations("original")
        representative_correlations = self._get_correlations("representative")

        correlation_errors = [{key_orig: abs(value_orig - value_repres) if key_orig == 'value' else value_orig for [[key_orig, value_orig], [key_repres, value_repres]] in zip(original.items(), representative.items())} for original, representative in zip(original_correlations, representative_correlations)] 
        
        logger.debug("correlation_errors: {}".format(correlation_errors))
        return correlation_errors