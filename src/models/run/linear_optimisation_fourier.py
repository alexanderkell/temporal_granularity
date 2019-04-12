from pathlib import Path
import sys
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))
import logging
import pandas as pd
import numpy as np
from src.models.optimisation_algorithms.fourier_resample.fourier_resample import fourier_resample
from src.models.manipulations.approximations import ApproximateData
from src.models.manipulations.duration_curves import get_ldc, get_rdc
from src.metrics.metrics import Metrics
from scipy.signal import resample
from scipy.optimize import minimize, brute

logger = logging.getLogger(__name__)


def import_data():
    project_dir = Path("__file__").resolve().parents[1]
    project_dir

    pv_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
    onshore_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
    load_data = pd.read_csv(
        '{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv'.format(project_dir))
    data = [pv_data, onshore_data, load_data]
    return data


def calculate_error_metrics(x0):
    number_day = 61

    total_metrics = []

    # resamples = np.ceil(365 / number_day)
    original = []
    approximated = []
    original_rdc = []
    approximated_rdc = []
    for dat in data:

        data_filtered = dat[(dat.datetime > "2015")
                            & (dat.datetime < "2016")]
        dat.fillna(method='ffill', inplace=True)
        data_filtered.fillna(method='ffill', inplace=True)
        # data_resampled = fourier_resample(
        #     data_filtered.capacity_factor, "capacity_factor", number_day, resamples)

        data_resampled = resample(
            data_filtered.capacity_factor, number_day * 24)

        x0 = np.array(x0).astype(int)
        logger.info("x0: {}".format(x0))

        data_resampled = np.repeat(
            data_resampled.reshape(12, -1), x0, axis=0).flatten()[:8760]

        data_resampled = pd.DataFrame({"capacity_factor": data_resampled})

        # data_sampled_ldc = get_ldc(
        # data_resampled, 'capacity_factor', index_name="index_for_year")
        # data_sampled_ldc['type'] = "sampled"
        data_sampled_ldc = get_ldc(
            data_resampled, 'capacity_factor', index_name="index_for_year")

        approximated.append(data_sampled_ldc)

        data_ldc = get_ldc(data_filtered, "capacity_factor",
                           index_name="index_for_year")
        # data_ldc['type'] = "actual"
        original.append(data_ldc)

        data_sampled_rdc = get_rdc(
            data_resampled, 'capacity_factor', index_name="index_for_year")

        approximated_rdc.append(data_sampled_rdc)
        data_ldc = get_rdc(data_filtered, "capacity_factor",
                           index_name="index_for_year")
        original_rdc.append(data_ldc)

    metrics_calculator = Metrics(original[0], approximated[0], original[1],
                                 approximated[1], original[2], approximated[2], "dc")

    metrics = metrics_calculator.get_mean_error_metrics()

    metrics_calculator_ldc = Metrics(original_rdc[0], approximated_rdc[0], original_rdc[1],
                                     approximated_rdc[1], original_rdc[2], approximated_rdc[2], "rdc")

    metrics_ldc = metrics_calculator_ldc.get_mean_error_metrics()

    metrics['day'] = number_day
    metrics_ldc['day'] = number_day
    total_metrics.append(metrics)
    total_metrics.append(metrics_ldc)
    total_metrics_df = pd.concat(total_metrics).reset_index()

    logger.info("total_metrics_df: {}".format(total_metrics_df))
    sum = np.float64(total_metrics_df.value.sum())
    logger.info("sum: {}".format(sum))
    return sum


def linear_optimisation():
    error_metrics = calculate_error_metrics(61, k)
    logger.debug("error_metrics: {}".format(error_metrics))


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    data = import_data()

    # for number_day in [4, 8, 12, 24, 48, 61]:

    # fun = calculate_error_metrics(61, x)

    cons = [{'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: x[3]},
            {'type': 'ineq', 'fun': lambda x: x[4]},
            {'type': 'ineq', 'fun': lambda x: x[5]},
            {'type': 'ineq', 'fun': lambda x: x[6]},
            {'type': 'ineq', 'fun': lambda x: x[7]},
            {'type': 'ineq', 'fun': lambda x: x[8]},
            {'type': 'ineq', 'fun': lambda x: x[9]},
            {'type': 'ineq', 'fun': lambda x: x[10]},
            {'type': 'ineq', 'fun': lambda x: x[11]}]

    # x = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    # result = minimize(calculate_error_metrics, x0=x, constraints=cons)
    x = (slice(0, 9, 1),) * 12
    result = brute(calculate_error_metrics, ranges=x,
                   disp=True, finish=None, full_output=True)

    # total_metrics_df = pd.concat(total_metrics).reset_index()
    # logger.debug("total_metrics_df: {}".format(total_metrics_df))
