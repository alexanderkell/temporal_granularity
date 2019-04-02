import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd
from src.metrics.single_metrics import SingleMetrics
import pytest

import numpy as np
import math 

import pytest

import logging

logging.basicConfig(level=logging.DEBUG)

class Test_Metrics:

    def test_nrmse(self):
        original_ldc = pd.DataFrame({"capacity_factor":[math.exp(i)-1 for i in np.arange(0, 0.70, 0.007)], "duration":[i for i in range(0,100,1)]})
        approximated_ldc = pd.DataFrame({"capacity_factor":[math.exp(i)-1+0.1 for i in np.arange(0, 0.70, 0.007)], "duration":[i for i in range(0,100,1)]})
        ldc_nrmse = SingleMetrics(original_ldc, approximated_ldc).nrmse()

        assert ldc_nrmse == pytest.approx(0.1, rel=0.1)

    def test_rae(self):
        original_ldc = pd.DataFrame({"capacity_factor":[math.exp(i)-1 for i in np.arange(0, 0.70, 0.007)], "duration":[i for i in range(0,100,1)]})
        approximated_ldc = pd.DataFrame({"capacity_factor":[math.exp(i)-1+0.1 for i in np.arange(0, 0.70, 0.007)], "duration":[i for i in range(0,100,1)]})
        ldc_nrmse = SingleMetrics(original_ldc, approximated_ldc).nrmse()

        assert ldc_nrmse == pytest.approx(0.1, rel=0.1)

    # def test_get_correlation(self):
    #     time_series_1 = pd.DataFrame({"capacity_factor":[1,0,1,0,1,0,1], "duration":[1,2,3,4,5,6,7]})
    #     time_series_2 = pd.DataFrame({"capacity_factor":[1,0,1,0,1,0,1], "duration":[1,2,3,4,5,6,7]})
    #     correlation = SingleMetrics(time_series_1, time_series_2).calculate_correlation()
    #     assert correlation == 1

    #     time_series_1 = pd.DataFrame({"capacity_factor":[0,1,0,1,0,1,0], "duration":[1,2,3,4,5,6,7]})
    #     time_series_2 = pd.DataFrame({"capacity_factor":[1,0,1,0,1,0,1], "duration":[1,2,3,4,5,6,7]})
    #     correlation = SingleMetrics(time_series_1, time_series_2, 20).calculate_correlation()
    #     assert correlation == -1