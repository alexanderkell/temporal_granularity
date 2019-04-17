import logging
from temporal_granularity.src.metrics.metrics import Metrics
from pandas.util.testing import assert_frame_equal
import pandas as pd
import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))


logging.basicConfig(level=logging.DEBUG)


class Test_Metrics:

    def test_all_nrmse(self):
        original_solar = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime": [1, 2, 3, 4, 5, 6, 7]})

        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, "dc")._get_nrmse()

        expected_nrmse = [{'metric': 'nrmse dc', 'series_type': 'solar', 'value': 16.666666666666668}, {'metric': 'nrmse dc',
                                                                                                        'series_type': 'wind', 'value': 0.0}, {'metric': 'nrmse dc', 'series_type': 'load', 'value': 8.33333333333334}]
        assert all_nrmse == expected_nrmse

    def test_all_rae(self):
        original_solar = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})

        all_nrmse = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, "dc")._get_rae()

        expected_nrmse = [{'metric': 'rae dc', 'series_type': 'solar', 'value': 7.142857142857138}, {'metric': 'rae dc',
                                                                                                     'series_type': 'wind', 'value': 0.0}, {'metric': 'rae dc', 'series_type': 'load', 'value': 3.5714285714285796}]
        assert all_nrmse == expected_nrmse

    def test_all_correlations(self):
        original_solar = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "datetime": [1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "datetime": [1, 2, 3, 4, 5, 6, 7]})

        metrics_calculator = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, "dc")
        original_correlations = metrics_calculator._get_correlations("original")

        expected_result = [{'metric': "correlation", "series_type": "solar-wind", "value": 1.0},
                           {'metric': "correlation", "series_type": "solar-load", "value": 1.0},
                           {'metric': "correlation", "series_type": "wind-load", "value": 1.0}]

        assert original_correlations == expected_result

        expected_result = [{'metric': "correlation", "series_type": "solar-wind", "value": 1.0}, {'metric': "correlation",
                                                                                                  "series_type": "solar-load", "value": 1.0}, {'metric': "correlation", "series_type": "wind-load", "value": 1.0}]

        representative_correlations = metrics_calculator._get_correlations("representative")

        assert representative_correlations == expected_result

    def test_get_error_metrics(self):
        original_solar = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_solar = pd.DataFrame({"capacity_factor": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        original_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_wind = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        original_load = pd.DataFrame({"capacity_factor": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})
        representative_load = pd.DataFrame({"capacity_factor": [1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65], "index_for_year": [1, 2, 3, 4, 5, 6, 7]})

        error_metrics = Metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, "dc").get_error_metrics()

        expected_result = pd.DataFrame({'metric': ['nrmse dc', 'nrmse dc', 'nrmse dc', 'rae dc', 'rae dc', 'rae dc', 'correlation', 'correlation', 'correlation'], 'series_type': [
                                       'solar', 'wind', 'load', 'solar', 'wind', 'load', "solar-wind", "solar-load", "wind-load"], 'value': [16.666666666666669, 0.0, 8.333333333333338, 7.142857142857155, 0.0, 3.57142857142855, 0, 0, 0]})

        assert_frame_equal(error_metrics, expected_result)
