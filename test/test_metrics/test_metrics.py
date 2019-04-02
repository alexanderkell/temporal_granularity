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
        
        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load).get_nrmse()

        expected_nrmse = [{'metric': 'nrmse', 'series_type': 'solar', 'value': 0.16666666666666669}, {'metric': 'nrmse', 'series_type': 'wind', 'value': 0.0}, {'metric': 'nrmse', 'series_type': 'load', 'value': 0.08333333333333338}]
        assert all_nrmse == expected_nrmse

    def test_all_rae(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        
        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load).get_rae()

        expected_nrmse = [{'metric': 'rae', 'series_type': 'solar', 'value': 0.07142857142857155}, {'metric': 'rae', 'series_type': 'wind', 'value': 0.0}, {'metric': 'rae', 'series_type': 'load', 'value': 0.0357142857142855}]
        assert all_nrmse == expected_nrmse

    def test_all_correlations(self):
        original_solar = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor":[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1,2,3,4,5,6,7]})
        representative_wind = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor":[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor":[1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime":[1, 2, 3, 4, 5, 6, 7]})
        
        all_correlations = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load).get_correlations()

        expected_result = [{'metric':"correlation", "series_type": "solar-wind-original", "value":1},
         {'metric':"correlation", "series_type": "solar-load-original", "value":1},
         {'metric':"correlation", "series_type": "wind-load-original", "value":1},
         {'metric':"correlation", "series_type": "solar-wind-representative", "value":1},
         {'metric':"correlation", "series_type": "solar-load-representative", "value":1},
         {'metric':"correlation", "series_type": "wind-load-representative", "value":1}]

        assert all_correlations == expected_result