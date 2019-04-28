from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

from src.metrics.multi_year_metrics import MultiYearMetrics
from src.models.manipulations.duration_curves import get_group_ldc
from src.models.optimisation_algorithms.genetic_algorithms.genetic_algorithm import GeneticAlgorithm

import pandas as pd

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    project_dir = Path("__file__").resolve().parents[1]

    # Import original data
    pv_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    wind_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_original = pd.read_csv(
        '{}/temporal_granularity/data/processed/demand/load_NG/load_processed_normalised.csv'.format(project_dir))

    onshore_wide = pd.read_csv('{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))
    load_wide = pd.read_csv('{}/temporal_granularity/data/processed/data_grouped_by_day/load_NG_normalised_each_day.csv'.format(project_dir))
    pv_wide = pd.read_csv('{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))


    results = []
    year_start = 2006
    year_end = 2010

    pv_wide = pv_wide[(pd.to_datetime(pv_wide['date']) >= str(year_start)) & (pd.to_datetime(pv_wide['date']) < str(year_end))]
    onshore_wide = onshore_wide[(pd.to_datetime(onshore_wide['date']) >= str(year_start)) & (pd.to_datetime(onshore_wide['date']) < str(year_end))]
    load_wide = load_wide[(pd.to_datetime(load_wide['date']) >= str(year_start)) & (pd.to_datetime(load_wide['date']) < str(year_end))]

    genetic_calculator = GeneticAlgorithm(pv_original, wind_original, load_original, onshore_wide, load_wide, pv_wide, year_end)
    toolbox = genetic_calculator.initialise()
    genetic_calculator.run_genetic_algorithm(toolbox)
    