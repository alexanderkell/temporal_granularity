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

    data = pd.read_csv(input_filepath)
    data.datetime = pd.to_datetime(data.datetime)
    data['date'] = data['datetime'].dt.date

    data['hour'] = data['datetime'].dt.hour
    each_day = data.pivot(index='date', columns='hour',
                          values='capacity_factor')

    each_day = each_day.dropna()
    logger.info(each_day.head())
    each_day.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]

    # main('{}/temporal_granularity/data/raw/resources/ninja_pv_country_GB_sarah_nuts-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    main('{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

    main('{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/data_grouped_by_day/load_normalised_each_day.csv'.format(project_dir))

    main('{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/data_grouped_by_day/onshore_each_day.csv'.format(project_dir))

    main('{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/data_grouped_by_day/offshore_each_day.csv'.format(project_dir))
