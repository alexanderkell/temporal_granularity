# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)


def main(input_filepath, output_filepath, output_filepath2):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = pd.read_csv(input_filepath)

    data = data.drop(columns="id")
    data = data.rename(columns={" timestamp": "datetime",
                                " demand": "capacity_factor"})

    data.datetime = pd.to_datetime(data.datetime)

    data = data.set_index("datetime")
    data = data.capacity_factor.resample("1h").mean()

    data = data.reset_index()


    data.to_csv(output_filepath)

    normalised_data = data.copy()


    logger.info(normalised_data.head())
    values = normalised_data.capacity_factor.values
    values = values.reshape((len(values), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)
    normalised_data.capacity_factor = normalized

    normalised_data.to_csv(output_filepath2)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]

    # main('{}/temporal_granularity/data/raw/resources/ninja_pv_country_GB_sarah_nuts-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    main('{}/temporal_granularity/data/raw/resources/gridwatch (3).csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/demand/load_processed.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv'.format(project_dir))
