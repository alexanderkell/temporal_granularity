import sys
from pathlib import Path

project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))
# from src.models.env.som_env import SOMEnv
from src.models.env.k_means_env import KMeansEnv
import pandas as pd

onshore_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

onshore_data_np = onshore_data.reset_index().drop(
    columns=["date", 'index']).values

load_data = pd.read_csv(
    "{}/temporal_granularity/data/processed/data_grouped_by_day/load_normalised_each_day.csv".format(project_dir))

load_data_np = load_data.reset_index().drop(columns=["date", 'index']).values


# offshore_data = pd.read_csv(
# '{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir))
pv_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

pv_data_np = pv_data.reset_index().drop(columns=["date", 'index']).values

pv_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
onshore_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
load_data = pd.read_csv(
    "{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv".format(project_dir))


individual = [60]
x1 = 102 * [0]

individual.extend(x1)


# individual = [i for i in x]
env = KMeansEnv(pv_data_np, onshore_data_np, load_data_np,
                pv_data, onshore_data, load_data, round(individual[0] / 10) + 1)
result = env.step(individual[1:])
result = result[0], result[1], result[2]
print(result)
