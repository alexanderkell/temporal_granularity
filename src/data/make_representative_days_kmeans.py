from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.manipulations.approximations import ApproximateData


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    pv_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    onshore_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    offshore_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir)) 
    load_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/demand/load_processed.csv'.format(project_dir))
    load_original = load_original[(load_original.datetime >= "2016") & (load_original.datetime < "2017")].reset_index(drop=True)

    total_data = [pv_original, onshore_original, offshore_original, load_original]
    data_name = ["solar", 'onshore', 'offshore', 'load']

    number_of_days = 8

    data_stored = []
    for data, name in zip(total_data, data_name):
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data[(data.datetime >= "2006") & (data.datetime < "2017")]
        data = data[~((data.datetime.dt.month == 2) & (data.datetime.dt.day == 29))]

        approximator = ApproximateData(data, number_of_days)
        medoids, y_kmeans_df = approximator.cluster_medoids()
        cluster_weights = y_kmeans_df.groupby(["cluster"]).size().reset_index(name='counts')
        
        result = pd.merge(medoids, cluster_weights, on="cluster")
        if data_name in ["solar", 'onshore', 'offshore']:
            result['counts'] = result['counts'] / 11
        result['data_type'] = name
        data_stored.append(result)

    granularity_data = pd.concat(data_stored)
    granularity_data.reset_index(inplace=True)
    
    granularity_data.to_csv("/Users/b1017579/Documents/PhD/Projects/10. ELECSIM/elecsim/data/processed/multi_day_data/8_medoids.csv")
    granularity_data.to_csv("/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/granular_data/8_medoids.csv")