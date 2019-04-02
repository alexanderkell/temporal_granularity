import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.approximations import ApproximateData

project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))





logging.basicConfig(level=logging.DEBUG)

class Test_Approximations():
    pass
