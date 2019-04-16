import pandas as pd
from temporal_granularity.src.metrics.metrics import Metrics


def get_multi_year_metrics(original_solar, representative_solar, original_wind, representative_wind, original_load, representative_load, curve_type):
    representative_data = [representative_solar, representative_wind, representative_load]

    grouped = []
    for data in representative_data:
        grouped_data = data.groupby("year")
        grouped.append(grouped_data)

    
