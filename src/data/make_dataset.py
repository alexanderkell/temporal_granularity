# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
import os

# os.path.join(os.path.dirname(__file__))

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    wind = pd.read_csv(input_filepath)
    print(wind.head())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[2]
    

    main('{}/data/raw/ninja_wind_country_GB_current-merra-2_corrected (2).csv'.format(project_dir), '{}/data/processed/wind_processed.csv'.format(project_dir))
