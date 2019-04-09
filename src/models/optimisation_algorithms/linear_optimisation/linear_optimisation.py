import pandas as pd
from pathlib import Path
import numpy as np


if __name__ == "__main__":

    project_dir = Path("__file__").resolve().parents[1]
    project_dir

    pv_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    onshore_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_data = pd.read_csv(
        "{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv".format(project_dir))

    
