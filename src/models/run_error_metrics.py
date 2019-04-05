from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

from sklearn.preprocessing import MinMaxScaler
from src.metrics.metrics import Metrics
from src.models.approximations import ApproximateData
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logger.info("Starting")

    # onshore_data = pd.read_csv(
    # '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/resources/onshore_processed.csv')
    # load_data = pd.read_csv(
    # "/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/demand/load_processed_normalised.csv")
    # pv_data = pd.read_csv(
    # "/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/resources/pv_processed.csv")

    onshore_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_data = pd.read_csv(
        "{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv".format(project_dir))

    # offshore_data = pd.read_csv(
    # '{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir))
    pv_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))

    data = [pv_data, onshore_data, load_data]

    results = []

    original_ldcs = []
    approximated_ldcs = []

    original_rdcs = []
    approximated_rdcs = []
    for i in range(10):
        logger.info("Running iteration {}".format(i))
        for method in ['centroids', 'medoids']:
            logger.info("Approximating using {} method".format(method))
            for num_days in [4, 8, 12, 24, 48]:
                logger.info("Calculating using {} days".format(num_days))
                original_ldcs.clear()
                approximated_ldcs.clear()

                original_rdcs.clear()
                approximated_rdcs.clear()
                for dat in data:
                    approximator = ApproximateData(dat, num_days)

                    original_ldc = approximator.get_load_duration_curve(
                        year="2014")
                    original_ldcs.append(original_ldc)

                    medoids_approximation = approximator.get_approximated_ldc(
                        method)
                    approximated_ldcs.append(medoids_approximation)

                    original_rdc = approximator.get_ramp_duration_curve(
                        year="2014")
                    original_rdcs.append(original_rdc)

                    approximated_rdc = approximator.get_approximated_rdc(
                        method)
                    approximated_rdcs.append(approximated_rdc)

                rdc_metrics_calculator = Metrics(
                    original_rdcs[0], approximated_rdcs[0], original_rdcs[1],
                    approximated_rdcs[1], original_rdcs[2], approximated_rdcs[2], "rdc")
                mean_rdc_errors = rdc_metrics_calculator.get_mean_error_metrics()

                mean_rdc_errors['num_days'] = num_days
                mean_rdc_errors['method'] = method
                results.append(mean_rdc_errors)

                ldc_metrics_calculator = Metrics(
                    original_ldcs[0], approximated_ldcs[0], original_ldcs[1], approximated_ldcs[1], original_ldcs[2], approximated_ldcs[2], "dc")
                mean_ldc_errors = ldc_metrics_calculator.get_mean_error_metrics()

                mean_ldc_errors['num_days'] = num_days
                mean_ldc_errors['method'] = method
                results.append(mean_ldc_errors)

    results_dataframe = pd.concat(results)
    results_dataframe = results_dataframe.reset_index()

    logger.info("results_dataframe: {}".format(results_dataframe))
    logger.info("columns name: {}".format(results_dataframe.columns))

    # g = sns.FacetGrid(data=results_dataframe, col="metric", hue="method" ,sharey=False)
    # g.map(sns.factorplot, "num_days", "value")
    # g = sns.catplot(x="num_days", y="value", hue="method", col="metric", data=results_dataframe, kind="box", sharey=False)
    g = sns.catplot(x="num_days", y="value", hue="method", col="metric",
                    data=results_dataframe, kind="box", sharey=False)
    # g.add_legend()
    plt.savefig(
        "{}/temporal_granularity/reports/figures/average_metrics_centroids.png".format(project_dir))
