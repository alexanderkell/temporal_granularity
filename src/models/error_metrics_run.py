import logging
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

from src.models.approximations import ApproximateData
from src.metrics.metrics import Metrics

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger.info("Starting")

    onshore_data = pd.read_csv('{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    offshore_data = pd.read_csv('{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir))
    pv_data = pd.read_csv('{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))

    data = [onshore_data, offshore_data, pv_data]

    results = []
    original_ldcs = []
    approximated_ldcs = []
    for num_days in [2,4,8,12,24,48]:
        original_ldcs.clear()
        approximated_ldcs.clear()
        for dat in data:
            approximator = ApproximateData(dat, num_days)

            original_ldc = approximator.get_load_duration_curve(year="2014")
            original_ldcs.append(original_ldc)

            medoids_approximation = approximator.get_approximated_ldc("medoids")

            approximated_ldcs.append(medoids_approximation)

        metrics_calculator = Metrics(original_ldcs[0], approximated_ldcs[0], original_ldcs[1], approximated_ldcs[1], original_ldcs[2], approximated_ldcs[2])
        # error_metrics = metrics_calculator.get_error_metrics()
        mean_errors = metrics_calculator.get_mean_error_metrics()

        mean_errors['num_days'] = num_days
        results.append(mean_errors)
    results_dataframe = pd.concat(results)
    results_dataframe = results_dataframe.reset_index()
    # results_dataframe = results_dataframe.groupby("metric")

    logger.info("results_dataframe: {}".format(results_dataframe))
    logger.info("columns name: {}".format(results_dataframe.columns))

    # results_dataframe.plot()
    g = sns.FacetGrid(data=results_dataframe, col="metric", sharey=False)
    g.map(plt.scatter, "num_days", "value")
    g.add_legend()
    plt.savefig("results234543.png")

