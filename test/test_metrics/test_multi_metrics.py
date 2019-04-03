import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd
from src.metrics.multi_metrics import MultiMetrics
import pytest

import numpy as np
import math 

import pytest

import logging

logging.basicConfig(level=logging.DEBUG)

class Test_MultiMetrics():

    def test_combinations(self):
        solar = pd.DataFrame({"capacity_factor":[0,1,0,1,0,1,0], "datetime":[1,2,3,4,5,6,7]})
        wind = pd.DataFrame({"capacity_factor":[0,1,0,1,0,1,0], "datetime":[1,2,3,4,5,6,7]})
        load = pd.DataFrame({"capacity_factor":[1,0,1,0,1,0,1], "datetime":[1,2,3,4,5,6,7]})
        multi_metrics_calculator = MultiMetrics(solar, wind, load)
        correlations = multi_metrics_calculator.get_correlations()

        expected_result = [{"metric":"correlation", "series_type":"solar-wind", "value":1.0},
        {"metric":"correlation", "series_type":"solar-load", "value":-1.0},
        {"metric":"correlation", "series_type":"wind-load", "value":-1.0}]

        assert correlations == expected_result

        solar = pd.DataFrame({"capacity_factor":[0,1,0,1,0,1,0], "datetime":[1,2,3,4,5,6,7]})
        wind = pd.DataFrame({"capacity_factor":[0,1,0,1,0,1,0], "datetime":[1,2,3,4,5,6,7]})
        load = pd.DataFrame({"capacity_factor":[0.7,0.5,0.7,0.5,0.7,0.5,0.8], "datetime":[1,2,3,4,5,6,7]})
        multi_metrics_calculator = MultiMetrics(solar, wind, load)
        correlations = multi_metrics_calculator.get_correlations()

        expected_result = [{"metric":"correlation", "series_type":"solar-wind", "value":1.0},
        {"metric":"correlation", "series_type":"solar-load", "value":pytest.approx(-0.9594, rel=0.001)},
        {"metric":"correlation", "series_type":"wind-load", "value":pytest.approx(-0.9594, rel=0.001)}]

        assert correlations == expected_result