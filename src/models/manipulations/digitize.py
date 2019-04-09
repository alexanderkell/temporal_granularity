import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def approximate_curve(data, bin_number):

    binned = pd.cut(data.capacity_factor, bin_number)

    # bins = np.arange(1, len(data.datetime) / bin_number + 1)
    # logger.debug("bins: {}".format(bins))
    # digitized = np.digitize(data, bins)
    # bin_means = [data[digitized == i].mean()
    #  for i in range(1, len(bin_number))]

    return binned
