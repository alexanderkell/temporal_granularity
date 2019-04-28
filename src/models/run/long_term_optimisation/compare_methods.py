from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.metrics.multi_year_metrics import MultiYearMetrics
from src.models.manipulations.approximations import ApproximateData
from src.models.manipulations.duration_curves import get_group_ldc


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def cluster_data(pv_original, wind_original, load_original, method, n_clusters, year_start, year_end):
    # Get LDC's for medoids
    pv_ldc = ApproximateData(pv_original[(pv_original.datetime > str(year_start)) & (
        pv_original.datetime < str(int(year_end) + 1))], n_clusters).get_approximated_ldc(method)
    wind_ldc = ApproximateData(
        wind_original[(wind_original.datetime > str(year_start)) & (wind_original.datetime < str(year_end + 1))], n_clusters).get_approximated_ldc(method)
    load_ldc = ApproximateData(
        load_original[(load_original.datetime > str(year_start)) & (load_original.datetime < str(year_end + 1))], n_clusters).get_approximated_ldc(method)
    return pv_ldc, wind_ldc, load_ldc



def get_each_ldc(pv_original, wind_original, load_original):
    pv_original_ldcs = get_group_ldc(
        pv_original, "capacity_factor", "index_for_year")
    wind_original_ldcs = get_group_ldc(
        wind_original, "capacity_factor", "index_for_year")
    load_original_ldcs = get_group_ldc(
        load_original, "capacity_factor", "index_for_year")
    return pv_original_ldcs, wind_original_ldcs, load_original_ldcs

if __name__ == '__main__':
    project_dir = Path("__file__").resolve().parents[1]

    # Import original data
    pv_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    wind_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/demand/load_NG/load_processed_normalised.csv'.format(project_dir))

    results = []
    year_start = 2006
    year_end = 2011
    # for method in ['medoids', 'centroids', 'season_average']:
    for method in ['medoids', 'centroids']:
        for i in range(10):
            logger.info("Iteration number {}".format(i))
            for n_clusters in [1, 2, 4, 8, 16, 32, 48, 61]:
                # Get LDC's for medoids
                pv_ldc, wind_ldc, load_ldc = cluster_data(pv_original, wind_original, load_original, method, n_clusters, year_start, year_end)

                # Get grouped LDC's for original data
                pv_original_ldcs, wind_original_ldcs, load_original_ldcs = get_each_ldc(pv_original, wind_original, load_original)

                # Get multi-year error metrics
                multi_year_metrics_calculator = MultiYearMetrics(
                    pv_original_ldcs, pv_ldc, wind_original_ldcs, wind_ldc, load_original_ldcs, load_ldc, year_end)
                medoids_multi_year_metrics = multi_year_metrics_calculator.get_multi_year_metrics("dc")
                medoids_multi_year_metrics['method'] = method
                medoids_multi_year_metrics['n_clusters'] = n_clusters
                results.append(medoids_multi_year_metrics)
    results_dataframe = pd.concat(results)
    results_dataframe.to_csv("{}/temporal_granularity/reports/data/long_term_method_comparison/long_term_method_comparison.csv".format(project_dir))
    # g = sns.FacetGrid(data=results_dataframe.reset_index(), col="year")
    # g = g.map(sns.lineplot, x="n_clusters", y="value", hue="metric", marker=".")
    # g = sns.catplot(x="n_clusters", y="value", hue="method", col="metric", data=medoids_multi_year_metrics, kind="box", sharey=False)
    # g.savefig("{}/temporal_granularity/reports/figures/long_term_method_comparison/long_term_method_comparison_different_clusters.png".format(project_dir))