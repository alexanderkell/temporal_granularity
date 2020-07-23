import logging
from temporal_granularity.src.metrics.metrics import Metrics
import pandas as pd
import sys

logger = logging.getLogger(__name__)


class MultiYearMetrics():
    def __init__(self, original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, year_start):
        self.original_solar = original_solar
        self.representative_solar = representative_solar
        self.original_wind = original_wind
        self.representative_wind = representative_wind
        self.original_load = original_load
        self.representative_load = representative_load
        self.year_start = year_start

        self.original_data = [self.original_solar, self.original_wind, self.original_load]
        # self.original_data = self._filter_leap_day()
        self._filter_leap_day()
        self.filter_for_full_years()

        self.representative_data = [self.representative_solar, self.representative_wind, self.representative_load]

    def get_multi_year_average_metrics(self, curve_type):
        multi_year_metrics = self.get_multi_year_metrics(curve_type)
        mean_metrics = multi_year_metrics.groupby("metric").mean()
        return mean_metrics

    def get_multi_year_metrics(self, curve_type):
        grouped = self._group_list_dataframes()

        error_metrics = []
        for (name1, solar), (name2, wind), (name3, load) in zip(grouped[0], grouped[1], grouped[2]):
            error_result = Metrics(solar, self.representative_solar, wind, self.representative_wind, load, self.representative_load, curve_type).get_mean_error_metrics()
            error_result['year'] = solar.year.iloc[0]
            error_metrics.append(error_result)
        total_metrics = pd.concat(error_metrics)
        return total_metrics

    def _group_list_dataframes(self):
        grouped = []
        for data in self.original_data:
            data = data[data.year > self.year_start]
            grouped_data = data.groupby("year", as_index=False)
            grouped.append(grouped_data)
        return grouped

    def _filter_leap_day(self):
        filtered = []

        for data in self.original_data:
            data.datetime = pd.to_datetime(data.datetime)
            # filtered_leap_year = data
            filtered_leap_year = data[~((data.datetime.dt.month == 2) & (data.datetime.dt.day == 29))]
            filtered.append(filtered_leap_year)
        self.original_data = filtered
        # return filtered

    def filter_for_full_years(self):

        filtered = []
        for data in self.original_data:
            filtered_dat = data[(data.datetime >= "2006-01-01 00:00:00") & (data.datetime < "2017")]
            filtered.append(filtered_dat)
        self.original_data = filtered
