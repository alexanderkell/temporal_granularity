import numpy as np
import pandas as pd

from src.models.env.env import Env
from src.models.manipulations.approximations import ApproximateData
from src.metrics.metrics import Metrics
import logging

logger = logging.getLogger(__name__)


class KMeansEnv(Env):

    def __init__(self, solar, onshore, load, solar_df, onshore_df, load_df, n_clusters):
        super().__init__(solar, onshore, load, solar_df, onshore_df, load_df)

        self.n_clusters = n_clusters

        self.kmeans_objects = []
        self.ykmeans_df = []
        for data in self.df_data_list:
            data_approximator = ApproximateData(data, self.n_clusters)
            _, kmeans, _, y_kmeans_df = data_approximator.perform_k_means_clustering()
            self.kmeans_objects.append(kmeans)
            self.ykmeans_df.append(y_kmeans_df)

    def step(self, actions):

        representative_data = []
        original_data = []

        actions = np.array(actions).reshape(3, -1)

        for kmeans, y_kmeans_df, np_data, df_data, actions_each_dat in zip(self.kmeans_objects, self.ykmeans_df, self.np_data_list, self.df_data_list, actions):

            number_of_clusters = kmeans.cluster_centers_.shape[0]
            representative_days = []
            for cluster_number, action in zip(range(number_of_clusters), actions_each_dat):
                d = kmeans.transform(np_data)[:, 0]

                ind = np.argsort(d)[::][:100]

                sorted_by_distance = np_data[ind]
                temp_df = pd.DataFrame(sorted_by_distance[action])
                temp_df.insert(0, "cluster", cluster_number)
                logger.debug("temp_df: {}".format(temp_df))
                representative_days.append(temp_df)

            rep_days_data = pd.concat(representative_days).rename(
                columns={0: "capacity_factor"})
            logger.debug("rep_days_data: {}".format(rep_days_data))
            # rep_days_data = self.wide_to_long(rep_days_data)

            approximation_calc = ApproximateData(df_data, 4)
            original_days = approximation_calc.get_load_duration_curve(
                year="2014")

            original_data.append(original_days)

            representative_days_ldc = ApproximateData(df_data, 4).get_load_duration_curve(
                rep_days_data, y_kmeans_df)

            representative_data.append(representative_days_ldc)
        metrics_calculator = Metrics(original_data[0], representative_data[0], original_data[1],
                                     representative_data[1], original_data[2], representative_data[2], "dc")

        error_metrics = metrics_calculator.get_mean_error_metrics()
        nrmse = error_metrics.iloc[1].value
        rae = error_metrics.iloc[2].value
        correlation = error_metrics.iloc[0].value

        return nrmse, rae, correlation
