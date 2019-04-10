import numpy as np
import pandas as pd
from scipy import signal
import logging

logger = logging.getLogger(__name__)


def fourier_resample(array, value_name, number_of_days, number_of_repeats, total_size=8760):
    data_sampled = np.repeat(signal.resample(
        array, 24 * number_of_days), int(number_of_repeats))[:total_size]

    data_sampled = pd.DataFrame({value_name: data_sampled})
    return data_sampled