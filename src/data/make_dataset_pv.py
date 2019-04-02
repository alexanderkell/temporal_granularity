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

    solar_data = pd.read_csv(input_filepath.format(project_dir), skiprows=2)
    
    solar_global = solar_data[['time','GB_TOTAL']]

    solar_global = solar_global.rename(columns={"GB_TOTAL":"capacity_factor","time":"datetime"})
    solar_global.datetime = pd.to_datetime(solar_global.datetime)

    solar_global.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]
    
    # main('{}/temporal_granularity/data/raw/resources/ninja_pv_country_GB_sarah_nuts-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    main('{}/temporal_granularity/data/raw/resources/ninja_pv_country_GB_sarah_nuts-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
