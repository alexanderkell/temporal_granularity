from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.models.manipulations.approximations import ApproximateData
from src.models.manipulations.duration_curves import get_ldc, get_rdc
from src.models.optimisation_algorithms.fourier_resample.fourier_resample import fourier_resample
from src.metrics.metrics import Metrics
pd.set_option('display.max_rows', 400)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    project_dir = Path("__file__").resolve().parents[1]
    project_dir

    pv_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    onshore_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv'.format(project_dir))

    data = [pv_data, onshore_data, load_data]

    total_metrics = []
    for number_day in [4, 8, 12, 24, 48, 61]:
        resamples = np.ceil(365 / number_day)
        original = []
        approximated = []
        original_rdc = []
        approximated_rdc = []
        for dat in data:

            data_filtered = dat[(dat.datetime > "2015") &
                                (dat.datetime < "2016")]
            dat.fillna(method='ffill', inplace=True)
            data_filtered.fillna(method='ffill', inplace=True)
            data_resampled = fourier_resample(
                data_filtered.capacity_factor, "capacity_factor", number_day, resamples)

            # data_resampled = signal.resample(
            #     data_filtered.capacity_factor, 24 * number_day)

            # data_resampled = np.repeat(
            #     data_resampled.reshape(12, -1), resamples).flatten()[:8760]

            # data_resampled = pd.DataFrame({"capacity_factor": data_resampled})

            # data_sampled_ldc = get_ldc(
            # data_resampled, 'capacity_factor', index_name="index_for_year")
            # data_sampled_ldc['type'] = "sampled"
            data_sampled_ldc = get_ldc(
                data_resampled, 'capacity_factor', index_name="index_for_year")

            approximated.append(data_sampled_ldc)

            data_ldc = get_ldc(data_filtered, "capacity_factor",
                               index_name="index_for_year")
            # data_ldc['type'] = "actual"
            original.append(data_ldc)

            data_sampled_rdc = get_rdc(
                data_resampled, 'capacity_factor', index_name="index_for_year")

            approximated_rdc.append(data_sampled_rdc)
            data_ldc = get_rdc(data_filtered, "capacity_factor",
                               index_name="index_for_year")
            original_rdc.append(data_ldc)

        metrics_calculator = Metrics(original[0], approximated[0], original[1],
                                     approximated[1], original[2], approximated[2], "dc")

        metrics = metrics_calculator.get_mean_error_metrics()

        metrics_calculator_ldc = Metrics(original_rdc[0], approximated_rdc[0], original_rdc[1],
                                         approximated_rdc[1], original_rdc[2], approximated_rdc[2], "rdc")

        metrics_ldc = metrics_calculator_ldc.get_mean_error_metrics()

        metrics['day'] = number_day
        metrics_ldc['day'] = number_day
        total_metrics.append(metrics)
        total_metrics.append(metrics_ldc)

    total_metrics_df = pd.concat(total_metrics).reset_index()

    total_metrics_df.to_csv(
        '{}/temporal_granularity/data/processed/results/fourier/fourier_results.csv'.format(project_dir))
    logger.debug("total_metrics_df: {}".format(total_metrics_df))
    sns.lineplot(data=total_metrics_df, x="day", y="value", hue="metric")
    plt.savefig(
        "/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/src/models/run/100_runs_fourier_repeat.png")
    # joined_data = pd.concat([data_sampled_ldc, data_ldc], sort=True)

    # sns.lineplot(data=joined_data, hue='type',
    #              y='capacity_factor', x='level_0')
    # plt.savefig(
    #     '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/src/models/run/fourier_ldc.png')
    # plt.close()

    # data_sampled_rdc = get_rdc(data_resampled)
    # data_sampled_rdc['type'] = 'sampled'
    # data_original_rdc = get_rdc(data)
    # data_original_rdc['type'] = 'actual'

    # joined_rdc = pd.concat([data_sampled_rdc, data_original_rdc], sort=True)

    # sns.lineplot(data=joined_rdc, hue='type', x='level_0', y='diff')
    # plt.savefig(
    #     '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/src/models/run/fourier_rdc.png')
    # plt.close()

    # medoids_clustered_rdc = ApproximateData(
    #     data, number_of_days).get_approximated_rdc("medoids")

    # medoids_clustered_rdc['type'] = 'medoids'
    # medoids_clustered_rdc.drop(['index_for_year'], axis=1, inplace=True)

    # medoids_clustered_rdc = medoids_clustered_rdc.reset_index()

    # joined_rdc = pd.concat(
    #     [medoids_clustered_rdc, data_original_rdc], sort=True)
    # sns.lineplot(data=joined_rdc, hue='type', x='level_0', y='diff')
    # plt.savefig(
    #     '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/src/models/run/fourier_medoids_rdc.png')
    # plt.close()
