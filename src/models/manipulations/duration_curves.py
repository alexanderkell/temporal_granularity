import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_ldc(data, column_to_sort, index_name="level_0"):
    """Get load duration curve.

    Converts a dataframe of any type and returns a load duration curve. 

    :param data: Data to be manipulated
    :type data: dataframe
    :param colum_to_sort: Column name to be sorted
    :type data: str
    :param index_name: Index to be named
    :type data: str
    :return: load duration curve.
    :rtype: dataframe
    """
    data_ldc = data.sort_values(
        column_to_sort, ascending=False).reset_index().reset_index()
    data_ldc.rename(index=str, columns={'level_0': index_name}, inplace=True)
    return data_ldc


def get_rdc(data, column, index_name="level_0"):
    """Get ramp duration curve.

    Converts dataframe into ramp duration curve

    :param data: Data to be manipulated
    :type data: dataframe
    :param column: Column name for calculations
    :type column: str
    :param index_name: Name to call index
    :type index_name: str
    :return: [description]
    :rtype: [type]
    """

    data['diff'] = data[column].diff()

    data_rdc = data.sort_values(
        'diff', ascending=False).reset_index().reset_index()

    data_rdc.rename(index=str, columns={'level_0': index_name}, inplace=True)

    return data_rdc
