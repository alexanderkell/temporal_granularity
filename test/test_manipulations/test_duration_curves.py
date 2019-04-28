from src.models.manipulations.duration_curves import get_group_ldc
from pandas.util.testing import assert_frame_equal
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Test_DurationCurveCalculator():

    def test_get_group_ldc(self):
        data = pd.DataFrame({"datetime": ["2015-01-01 10:00", "2015-01-01 11:00", "2015-01-01 12:00", "2016-01-01 10:00", "2016-01-01 11:00", "2016-01-01 12:00"], "capacity_factor": [1, 2, 3, 4, 5, 6]})
        data.datetime = pd.to_datetime(data.datetime)
        grouped_ldc = get_group_ldc(data, "capacity_factor", "level_0")

        result_dataframe = pd.DataFrame({"level_0": [0, 1, 2, 0, 1, 2], "capacity_factor": [3, 2, 1, 6, 5, 4], "year": [2015, 2015, 2015, 2016, 2016, 2016]})
        assert_frame_equal(grouped_ldc[['level_0', 'capacity_factor', 'year']].reset_index(drop=True), result_dataframe)
