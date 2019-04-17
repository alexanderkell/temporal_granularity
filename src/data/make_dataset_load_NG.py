# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)


def main(data, output_filepath, output_filepath2):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = data[['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD', 'ND']]
    data['SETTLEMENT_PERIOD'] = data['SETTLEMENT_PERIOD'] * 30
    data['SETTLEMENT_PERIOD'] = pd.to_datetime(data['SETTLEMENT_PERIOD'], unit='m').dt.strftime('%H:%M')

    data['SETTLEMENT_DATE'] = data['SETTLEMENT_DATE'] + " " + data['SETTLEMENT_PERIOD']
    data.drop(inplace=True, columns=['SETTLEMENT_PERIOD'])

    data.rename(inplace=True, columns={"SETTLEMENT_DATE": "datetime", 'ND': 'capacity_factor'})

    data.datetime = pd.to_datetime(data.datetime)

    data = data.set_index("datetime")
    data.index = pd.to_datetime(data.index)

    data = data.capacity_factor.resample("h").mean()

    logger.debug("data: \n{}".format(data))
    data = data.reset_index()

    values = data.capacity_factor.values
    values = values.reshape((len(values), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    # normalize the dataset and print the first 5 rows
    normalized = scaler.transform(values)

    normalized_data = data.copy()
    normalized_data.capacity_factor = normalized

    logger.debug(normalized_data.head())

    data.to_csv(output_filepath)
    normalized_data.to_csv(output_filepath2)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path("__file__").resolve().parents[1]

    load_1 = pd.read_csv("{}/temporal_granularity/data/raw/resources/demand_nationalgrid/DemandData_2005-2010 (1).csv".format(project_dir))
    load_2 = pd.read_csv("{}/temporal_granularity/data/raw/resources/demand_nationalgrid/DemandData_2011-2016 (1).csv".format(project_dir))
    load_3 = pd.read_csv("{}/temporal_granularity/data/raw/resources/demand_nationalgrid/DemandData_2017 (2).csv".format(project_dir))
    load_4 = pd.read_csv("{}/temporal_granularity/data/raw/resources/demand_nationalgrid/DemandData_2018.csv".format(project_dir))

    load = pd.concat([load_1, load_2, load_3, load_4])

    # main('{}/temporal_granularity/data/raw/resources/ninja_pv_country_GB_sarah_nuts-2_corrected.csv'.format(project_dir), '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    main(load,
         '{}/temporal_granularity/data/processed/demand/load_NG/load_processed.csv'.format(project_dir),
         '{}/temporal_granularity/data/processed/demand/load_NG/load_processed_normalised.csv'.format(project_dir))
