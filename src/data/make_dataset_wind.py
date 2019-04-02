# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
import os

# os.path.join(os.path.dirname(__file__))

def main(input_filepath, output_filepath, dataset_string):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    wind_dat = pd.read_csv(input_filepath.format(project_dir), skiprows=2)
    
    wind_off_onshore = wind_dat.drop('national', axis=1)
    wind_off_onshore = wind_off_onshore[['time',dataset_string]]

    wind_off_onshore = wind_off_onshore.rename(index=str, columns={dataset_string:'capacity_factor'})
    print(wind_off_onshore.columns)
    wind_off_onshore['datetime']=wind_off_onshore.time
    print(wind_off_onshore.head())
    wind_off_onshore.datetime=pd.to_datetime(wind_off_onshore.datetime)
    wind_off_onshore[['datetime','capacity_factor']].to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]
    
    dataset_string = 'onshore'
    # main('{}/temporal_granularity/data/raw/resources/ninja_wind_country_GB_current-merra-2_corrected (2).csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/{}_processed.csv'.format(project_dir, dataset_string), dataset_string)
    main('{}/temporal_granularity/data/raw/resources/belgium/ninja_wind_country_BE_current-merra-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/{}_processed_BE.csv'.format(project_dir, dataset_string), dataset_string)
    
    dataset_string = 'offshore'
    main('{}/temporal_granularity/data/raw/resources/belgium/ninja_wind_country_BE_current-merra-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/{}_processed_BE.csv'.format(project_dir, dataset_string), dataset_string)
