"""
 Description: Wrapper class which calculates error metrics required utilising multi metrics and single metrics class.

 Created on Fri Apr 05 2019

 Copyright (c) 2019 Newcastle University
 License is MIT
 Email is alexander@kell.es
# """

import logging

import pandas as pd

from temporal_granularity.src.metrics.multi_metrics import MultiMetrics
from temporal_granularity.src.metrics.single_metrics import SingleMetrics

logger = logging.getLogger(__name__)


class Metrics:

    def __init__(self, original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, curve_type):
        """Calculate error metrics.

        Calculates error metrics for the curves constructor is initialised
        with. Can calculate NRMSE, RAE, and different in correlation.

        :param original_solar: Original solar capacity factor curve
        :type original_solar: Pandas dataframe
        :param representative_solar: Solar capacity factor curve which approximates original curve
        :type representative_solar: Pandas dataframe
        :param original_wind: Original wind capacity factor curve
        :type original_wind: Pandas dataframe
        :param representative_wind: Wind capacity factor curve which approximates original curve
        :type representative_wind: Pandas dataframe
        :param original_load: Original load curve
        :type original_load: Pandas dataframe
        :param representative_load: Load curve which approximates original load curve
        :type representative_load: Pandas dataframe
        :param curve_type: Name of curve to enable differentiation between types of curve
        :type curve_type: str
        """
        self.original_solar = original_solar
        self.representative_solar = representative_solar
        self.original_wind = original_wind
        self.representative_wind = representative_wind
        self.original_load = original_load
        self.representative_load = representative_load
        self.curve_type = curve_type

    def get_mean_error_metrics(self):
        """Get mean error metrics.

        Calculates the mean of the error metrics for the curves

        :return: error metrics in the form of rae, nrmse, and correlation
        :rtype: pandas dataframe
        """
        error_metrics = self.get_error_metrics()
        mean_errors = error_metrics.groupby("metric").value.mean().to_frame()
        return mean_errors

    def get_error_metrics(self):
        """Get error metrics for each curve type.

        Calculate the error metrics for each curve. Includes nrmse, rae,
        correlation

        :return: all error metrics
        :rtype: pandas dataframe
        """
        metrics = []

        all_nrmse = self._get_nrmse()
        metrics.extend(all_nrmse)
        if self.curve_type == "dc":
            all_rae = self._get_rae()
            all_corr = self._calculate_correlation_error()

            metrics.extend(all_rae)
            metrics.extend(all_corr)

        results = pd.DataFrame(metrics)
        return results

    def _get_nrmse(self):
        nrmse_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            # print(data_type)
            nrmse_result = {}
            solar_metrics = SingleMetrics(
                original[1], representative[1]).nrmse()
            nrmse_result.update({'series_type': data_type, 'metric': 'nrmse {}'.format(
                self.curve_type), 'value': solar_metrics})
            nrmse_results.append(nrmse_result)

        return nrmse_results

    def _get_rae(self):
        rae_results = []
        data_types = ['solar', 'wind', 'load']
        members = list(vars(self).items())
        for original, representative, data_type in zip(members[0::2], members[1::2], data_types):
            rae_result = {}
            rae_value = SingleMetrics(original[1], representative[1]).rae()
            rae_result.update({'series_type': data_type, 'metric': 'rae {}'.format(
                self.curve_type), 'value': rae_value})
            rae_results.append(rae_result)

        return rae_results

    def _get_correlations(self, data_type):

        all_correlations = []
        if data_type == "original":
            original_correlations = MultiMetrics(
                self.original_solar, self.original_wind, self.original_load).get_correlations()
            all_correlations.extend(original_correlations)
        elif data_type == "representative":
            representative_correlations = MultiMetrics(
                self.representative_solar, self.representative_wind, self.representative_load).get_correlations()
            all_correlations.extend(representative_correlations)
        else:
            raise ValueError(
                "data_type can not be {}. It must be original or representative".format(data_type))
        return all_correlations

    def _calculate_correlation_error(self):
        original_correlations = self._get_correlations("original")
        representative_correlations = self._get_correlations("representative")

        correlation_errors = [{key_orig: abs(value_orig - value_repres) if key_orig == 'value' else value_orig for [[key_orig, value_orig], [key_repres, value_repres]] in zip(
            original.items(), representative.items())} for original, representative in zip(original_correlations, representative_correlations)]

        return correlation_errors
