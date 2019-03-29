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

import logging

class approximate_data:

    def __init__(self, data_dir, data_type):
        self.data = self.import_data(data_dir)
        self.data_type = data_type
        

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

    def cluster_on_kmeans(self, n_clusters):

        each_day = self.long_to_wide_data()

        kmeans = self.get_kmeans(each_day, n_clusters)
        y_kmeans = kmeans.predict(each_day)

        cluster_centres = kmeans.cluster_centers_
        cluster_centres_df = pd.DataFrame(cluster_centres)
        cluster_centres_df['cluster'] = cluster_centres_df.index
        cluster_centres_df.reset_index()
        
        centres_df_long = pd.melt(cluster_centres_df, id_vars="cluster", value_vars=[0,         1,         2,         3,         4,         5,
               6,         7,         8,         9,        10,        11,
              12,        13,        14,        15,        16,        17,
              18,        19,        20,        21,        22,        23,])
        
        print(centres_df_long)
        return centres_df_long


    def cluster_medoids(self, n_clusters):

        each_day = self.long_to_wide_data()

        k_means = self.get_kmeans(each_day, n_clusters)
        y_kmeans = k_means.predict(each_day)

        each_day_w_kmeans = each_day.copy()
        each_day_w_kmeans['cluster'] = y_kmeans

        closest_data_points, _ = pairwise_distances_argmin_min(k_means.cluster_centers_, each_day)

        medoids = self.get_medoids(closest_data_points, each_day_w_kmeans)
            

        return medoids

    def import_data(self, data_dir):
        data = pd.read_csv(data_dir).drop('Unnamed: 0', axis=1)


        data.set_index('datetime', inplace=True)
        data.index=pd.to_datetime(data.index)

        data['date'] = data.index.date
        data['hour'] = data.index.hour
        data['year'] = data.index.year
        return data

        
    def get_medoids(self, closest_data_points, each_day_w_kmeans):
        medoids = pd.DataFrame([])
        for index in closest_data_points:
            closest_datapoint = pd.DataFrame(each_day_w_kmeans.iloc[index,:])
            closest_datapoint['cluster']=closest_datapoint.iloc[24,0]
            closest_datapoint = closest_datapoint.rename(columns={ closest_datapoint.columns[0]: "value" })
            closest_datapoint=closest_datapoint.drop('cluster', axis=0)
            medoids = medoids.append(closest_datapoint)
        return medoids


    def long_to_wide_data(self):
        date_hour = self.data.copy()


        each_day = date_hour.pivot(index='date', columns='hour', values='capacity_factor')
        each_day = each_day.dropna()

        return each_day

    def get_kmeans(self, each_day, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(each_day)
        return kmeans


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]

    data_manipulator = approximate_data('{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir), 'onshore')
    # data_manipulator.average_by_season()
    # data_manipulator.cluster_on_kmeans(4)
    data_manipulator.cluster_medoids(4)
