from pathlib import Path
import pandas as pd
from src.metrics.multi_year_metrics import MultiYearMetrics
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

project_dir = Path("__file__").resolve().parents[1]


@pytest.fixture
def define_multi_year_metrics():
    pv_original = pd.read_csv('{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    pv_representative = pd.DataFrame({"index_for_year": list(range(8760)), "capacity_factor": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] * 365})
    wind_representative = pd.DataFrame({"index_for_year": list(range(8760)), "capacity_factor": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] * 365})
    load_representative = pd.DataFrame({"index_for_year": list(range(8760)), "capacity_factor": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] * 365})
    wind_original = pd.read_csv('{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    # load_original = pd.read_csv('{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv'.format(project_dir))
    load_original = pd.read_csv('{}/temporal_granularity/data/processed/demand/load_NG/load_processed_normalised.csv'.format(project_dir))

    original_data = []
    for dat in [pv_original, wind_original, load_original]:
        dat.datetime = pd.to_datetime(dat.datetime)
        dat['year'] = dat.datetime.dt.year
        original_data.append(dat)

    multi_year_metrics_calc = MultiYearMetrics(original_data[0], pv_representative, original_data[1], wind_representative, original_data[2], load_representative)
    yield multi_year_metrics_calc


class Test_MultiYearMetrics:

    def test_group_list_dataframes(self, define_multi_year_metrics):

        grouped_dfs = define_multi_year_metrics._group_list_dataframes()
        assert len(grouped_dfs) == 3
        assert list(grouped_dfs[0].groups.keys()) == list(range(1980, 2017))
        assert list(grouped_dfs[1].groups.keys()) == list(range(1980, 2017))
        assert list(grouped_dfs[2].groups.keys()) == list(range(2005, 2019))

    def test_get_multi_year_metrics(self, define_multi_year_metrics):

        result_errors = define_multi_year_metrics.get_multi_year_metrics("dc")

    def test_get_multi_year_average_metrics(self, define_multi_year_metrics):
        mean_errors = define_multi_year_metrics.get_multi_year_average_metrics("dc")
        logger.debug(mean_errors)
