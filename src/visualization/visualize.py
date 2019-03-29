import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
import os
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))
project_dir = Path("__file__").resolve().parents[1]


from src.models.approximations import ApproximateData
import pandas as pd
pd.set_option('display.max_rows', 500)
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import cycle

def plot_orginal_and_approximated_ldc(data_dir, approximation_method, n_clusters, data_type):

    data_manipulator = ApproximateData(data_dir, data_type)
    original = data_manipulator.get_load_duration_curve()

    if approximation_method == 'medoids':
        centres, y_kmeans = data_manipulator.cluster_medoids(n_clusters)
        data = data_manipulator.get_load_duration_curve(centres, y_kmeans)
    elif approximation_method == "centroids":
        centres, y_kmeans = data_manipulator.kmeans_centroids(n_clusters)
        data = data_manipulator.get_load_duration_curve(centres, y_kmeans)
    elif approximation_method == "season_average":
        centres = data_manipulator.average_by_season()
        data = data_manipulator.get_load_duration_curve(centres)
    else:
        raise ValueError("approximation method can not equal {}, must be medoids, centroids or season_average".format(approximation_method))

    fig, ax = plt.subplots()
    sns.lineplot(data=original, hue='year', x='index_for_year', y='capacity_factor', ax=ax)
    sns.lineplot(data=data, y='capacity_factor', x='index_for_year')
    plt.xlabel("Hours at capacity factor (h)")
    plt.ylabel("Capacity Factor")
    plt.title("Original load duration curve for each year\n for {} and {} with {} clusters".format(data_type, approximation_method, n_clusters))


    directory = "{}/temporal_granularity/src/visualization/figures/{}".format(project_dir, data_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_dir = "{}/{}-clusters-{}.png".format(directory, approximation_method, n_clusters)
    plt.savefig(image_dir)

    plt.close()



if __name__ == "__main__":
    onshore_data = '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/resources/onshore_processed.csv'
    offshore_data = '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/resources/offshore_processed.csv'
    pv_data = '/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/data/processed/resources/pv_processed.csv'

    data = [onshore_data, offshore_data, pv_data, onshore_data, offshore_data, pv_data]
    resources = ['onshore', 'offshore', 'photovoltaic','onshore', 'offshore', 'photovoltaic']
    method = ['centroids', 'medoids']

    for data, resource, method in zip(data, resources, cycle(method)):
        for n_clusters in range(1,10):
            plot_orginal_and_approximated_ldc(data, method, n_clusters, resource)

        
