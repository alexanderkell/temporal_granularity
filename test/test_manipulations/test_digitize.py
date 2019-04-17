from src.models.manipulations.digitize import approximate_curve
import pandas as pd
import sys
from pathlib import Path
import logging
import pytest
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.util.testing import assert_frame_equal

project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


class Test_Digitize:

    def test_digitize(self):
        ldc = pd.DataFrame(
            {"datetime": [0, 1, 2, 3, 4, 5], "capacity_factor": [1, 2, 3, 4, 5, 6]})

        digitized_ldc = approximate_curve(ldc, 3).to_frame().reset_index()

        # logger.info(digitized_ldc)
        digitized_ldc.capacity_factor = digitized_ldc['capacity_factor'].apply(
            lambda x: x.right)

        sns.lineplot(data=digitized_ldc, x='index', y='capacity_factor')
        sns.lineplot(data=ldc, x='datetime', y='capacity_factor')
        sns.savefig("actual.png")

        logger.info(digitized_ldc.capacity_factor)
        # logger.debug("digitized_ldc: {}".format(digitized_ldc(3.5)))
        assert_frame_equal(digitized_ldc, pd.DataFrame({"datetime": [1, 2, 3, 4, 5, 6], "capacity_factor": [2, 2, 4, 4, 6, 6]}))
