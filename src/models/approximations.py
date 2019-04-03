import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import signal
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ApproximateData:

    def __init__(self, data, n_days):
        self.data = self._import_data(data)
        self.n_days = n_days
        

    def get_approximated_ldc(self, approximation_method):
        if approximation_method == 'medoids':
            centres, y_kmeans = self.cluster_medoids()
            data = self.get_load_duration_curve(centres, y_kmeans)
        elif approximation_method == "centroids":
            centres, y_kmeans = self.kmeans_centroids()
            data = self.get_load_duration_curve(centres, y_kmeans)
        elif approximation_method == "season_average":
            centres = self.average_by_season()
            data = self.get_load_duration_curve(centres)
        else:
            raise ValueError("approximation method can not equal {}, must be medoids, centroids or season_average".format(approximation_method))
        return data

    def average_by_season(self):
        bins = [0, 70, 163, 245, 358, 366]
        labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter1']
        doy = self.data.index.dayofyear
        data_season = self.data.copy()
        data_season['season'] = pd.cut(doy + 11 - 366*(doy > 355), bins=bins, labels=labels)
        data_season.loc[data_season['season']=='Winter1','season']='Winter'
        
        data_hour_day = data_season.copy()
        data_hour_day['hour'] = self.data.index.hour
        data_hour_day['day'] = self.data.index.dayofweek
        
        average_day_df = data_hour_day.groupby(['hour', 'season']).capacity_factor.mean()
        average_day_df = pd.DataFrame(average_day_df)
        average_day_df = average_day_df.reset_index()

        return average_day_df

    def kmeans_centroids(self):

        each_day = self._long_to_wide_data()

        kmeans = self._get_kmeans(each_day, self.n_days)
        y_kmeans = kmeans.predict(each_day)
        y_kmeans_df = pd.DataFrame(y_kmeans)
        y_kmeans_df = y_kmeans_df.rename(columns={y_kmeans_df.columns[0]:'cluster'})

        cluster_centres = kmeans.cluster_centers_
        cluster_centres_df = pd.DataFrame(cluster_centres)
        cluster_centres_df['cluster'] = cluster_centres_df.index
        cluster_centres_df.reset_index()
        
        centres_df_long = pd.melt(cluster_centres_df, id_vars="cluster", value_vars=[0,         1,         2,         3,         4,         5,
               6,         7,         8,         9,        10,        11,
              12,        13,        14,        15,        16,        17,
              18,        19,        20,        21,        22,        23,])
        
        centres_df_long = centres_df_long.rename(columns={"value":"capacity_factor"})

        return centres_df_long, y_kmeans_df


    def cluster_medoids(self):

        each_day = self._long_to_wide_data()

        k_means = self._get_kmeans(each_day)
        y_kmeans = k_means.predict(each_day)

        y_kmeans_df = pd.DataFrame(y_kmeans)
        y_kmeans_df = y_kmeans_df.rename(columns={y_kmeans_df.columns[0]:'cluster'})

        each_day_w_kmeans = each_day.copy()
        each_day_w_kmeans['cluster'] = y_kmeans

        closest_data_points, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, each_day)

        medoids = self._get_medoids(closest_data_points, each_day_w_kmeans)

        return medoids, y_kmeans_df

    
    def get_load_duration_curve(self, data=None, clusters=None, year=None):
        if data is None:
            data_each_year = self.data.copy()
            data_each_year= self.data.reset_index()
            data_each_year['year'] = data_each_year.datetime.dt.year
            if year is None:    
                data_each_year = data_each_year.groupby('year').apply(lambda x: x.sort_values('capacity_factor', ascending=False).reset_index().reset_index())
            else:
                data_each_year = data_each_year[(data_each_year.datetime>=year) & (data_each_year.datetime<str(int(year)+1))]
                data_each_year = data_each_year.sort_values('capacity_factor', ascending=False).reset_index().reset_index()
        elif clusters is None:
            data_each_year = self._scale_data_to_seasons(data)
            data_each_year = data_each_year.sort_values('capacity_factor', ascending=False).reset_index().reset_index()
        else:
            data_each_year = self._scale_data_to_clusters(data, clusters)
            data_each_year = data_each_year.sort_values('capacity_factor', ascending=False).reset_index()
            data_each_year = data_each_year.drop('level_0', axis=1).reset_index()

        data_each_year = data_each_year.rename(columns={"level_0":"index_for_year"})
        return data_each_year

    def get_ramp_curve(self, data):
        data['diff'] = data.diff()['capacity_factor']
        data_sorted = data.sort_values('diff', ascending=False).reset_index().reset_index()

        return data_sorted

    def _scale_data_to_seasons(self, data):
        bins = [0, 70, 163, 245, 358, 366]
        days_per_season = [x - bins[i - 1] for i, x in enumerate(bins)][1:]
        days_per_season[0] = days_per_season[0]+days_per_season[-1]
        del days_per_season[-1]
        labels=['Winter', 'Spring', 'Summer', 'Autumn']
        season_scaling = {'season':labels, 'days':days_per_season}
        season_scaling_df = pd.DataFrame(season_scaling)
        season_scaling_df.head()


        average_day_scaled = data.merge(season_scaling_df, on='season')

        average_day_scaled = average_day_scaled.reindex(average_day_scaled.index.repeat(average_day_scaled.days))
        
        return average_day_scaled

    def _scale_data_to_clusters(self, data, clusters):
        clusters = clusters.reset_index()
        cluster_weights = clusters.groupby('cluster').count()
        scaled_data = data.merge(cluster_weights, on='cluster') #[['cluster','capacity_factor','year']]
        scaled_data['index'] = scaled_data['index']/37

        scaled_data['index'] = np.ceil(scaled_data['index']).astype(int)
        # scaled_data['index'] = scaled_data['index'].round(decimals=0)
        scaled_data = scaled_data.reindex(scaled_data.index.repeat(scaled_data['index']))
        
        scaled_data = scaled_data[:8760]

        return scaled_data

    def _import_data(self, data):
        data = data.drop('Unnamed: 0', axis=1)

        data.set_index('datetime', inplace=True)
        data.index=pd.to_datetime(data.index)

        data['date'] = data.index.date
        data['hour'] = data.index.hour
        data['year'] = data.index.year
        return data

        
    def _get_medoids(self, closest_data_points, each_day_w_kmeans):
        medoids = pd.DataFrame([])
        for index in closest_data_points:
            closest_datapoint = pd.DataFrame(each_day_w_kmeans.iloc[index,:])
            closest_datapoint['cluster']=closest_datapoint.iloc[24,0]
            closest_datapoint = closest_datapoint.rename(columns={ closest_datapoint.columns[0]: "capacity_factor" })
            closest_datapoint=closest_datapoint.drop('cluster', axis=0)
            medoids = medoids.append(closest_datapoint)
        return medoids


    def _long_to_wide_data(self):
        date_hour = self.data.copy()


        each_day = date_hour.pivot(index='date', columns='hour', values='capacity_factor')
        each_day = each_day.dropna()

        return each_day

    def _get_kmeans(self, each_day):
        kmeans = KMeans(n_clusters=self.n_days)
        kmeans.fit(each_day)
        return kmeans


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]

    data = pd.read_csv('{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    data_manipulator = ApproximateData(data, 'onshore')
    # data_manipulator.average_by_season()
    data_manipulator.kmeans_centroids(4)
    # data_manipulator.cluster_medoids(4)
    # print(data_manipulator.get_load_duration_curve())
