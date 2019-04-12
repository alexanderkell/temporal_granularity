import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    fourier = pd.read_csv(
        '{}/temporal_granularity/data/processed/results/fourier/fourier_results.csv'.format(project_dir))

    fourier['method'] = "fourier"
    fourier.rename(columns={'day': "num_days"}, inplace=True)
    k_means = pd.read_csv(
        '{}/temporal_granularity/data/processed/results/k_means/run_error_metrics.csv'.format(project_dir))

    joined = pd.concat([k_means, fourier])
    logger.debug("joined: \n{}".format(joined))
    g = sns.factorplot(x="num_days", y="value", hue="method", col="metric",
                       data=joined, kind="bar", sharey=False)
    # g = sns.catplot(x="num_days", y="value", hue="method", col="metric",
    #                 data=joined, kind="box", sharey=False)
    plt.savefig(
        "{}/temporal_granularity/src/visualization/figures/method_comparison/k_means_fourier_results.png".format(project_dir))
