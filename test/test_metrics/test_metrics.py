import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd
from pandas.util.testing import assert_frame_equal
from src.metrics.metrics import Metrics
import pytest

import numpy as np
import math 

import pytest

import logging

logging.basicConfig(level=logging.DEBUG)

class Test_Metrics:

    def test_all_nrmse(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        
        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load)._get_nrmse()

        expected_nrmse = [{'metric': 'nrmse', 'series_type': 'solar', 'value': 0.16666666666666669}, {'metric': 'nrmse', 'series_type': 'wind', 'value': 0.0}, {'metric': 'nrmse', 'series_type': 'load', 'value': 0.08333333333333338}]
        assert all_nrmse == expected_nrmse

    def test_all_rae(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        
        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load)._get_rae()

        expected_nrmse = [{'metric': 'rae', 'series_type': 'solar', 'value': 0.07142857142857155}, {'metric': 'rae', 'series_type': 'wind', 'value': 0.0}, {'metric': 'rae', 'series_type': 'load', 'value': 0.0357142857142855}]
        assert all_nrmse == expected_nrmse

    def test_all_correlations(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        
        all_correlations = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load)._get_correlations()

        expected_result = [{'metric':"correlation", "series_type": "solar-wind-original", "value":1},
         {'metric':"correlation", "series_type": "solar-load-original", "value":1},
         {'metric':"correlation", "series_type": "wind-load-original", "value":1},
         {'metric':"correlation", "series_type": "solar-wind-representative", "value":1},
         {'metric':"correlation", "series_type": "solar-load-representative", "value":1},
         {'metric':"correlation", "series_type": "wind-load-representative", "value":1}]

        assert all_correlations == expected_result

    def test_get_error_metrics(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})

        error_metrics = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load).get_error_metrics()

        expected_result = pd.DataFrame({'metric': ['nrmse','nrmse','nrmse', 'rae', 'rae', 'rae','correlation','correlation','correlation','correlation','correlation','correlation'], 'series_type': ['solar', 'wind', 'load','solar', 'wind', 'load',"solar-wind-original","solar-load-original", "wind-load-original", "solar-wind-representative", "solar-load-representative", "wind-load-representative"], 'value': [0.16666666666666669, 0.0, 0.08333333333333338, 0.07142857142857155, 0.0, 0.0357142857142855, 1, 1, 1, 1, 1, 1]})

        assert_frame_equal(error_metrics, expected_result)
        # assert error_metrics.sort_index().sort_index(axis=1) == expected_result.sort_index().sort_index(axis=1)