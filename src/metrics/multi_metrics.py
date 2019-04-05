import logging
from scipy import stats

from itertools import combinations

logger = logging.getLogger(__name__)


class MultiMetrics:
    """Calculates error metrics which require more than one comparison.

    Calculates error metrics which require more than one comparison, for example correlation which
    compares every curve against every other curve

    :return: MultiMetrics object
    :rtype: MultiMetrics
    """

    def __init__(self, solar, wind, load):
        """Instantiate MultiMetrics object with solar, wind and load curve.

        Get metrics which require the comparison of solar wind and load curves

        :param solar: Any curve which represents solar
        :type solar: pandas dataframe
        :param wind: Any curve which represents wind
        :type wind: pandas dataframe
        :param load: Any curve which represents load
        :type load: pandas dataframe
        """

        self.solar = solar
        self.wind = wind
        self.load = load

    def get_correlations(self):
        """Return correlations for each of the curves.

        Calculates the pearson correlation coefficient of each of the curves against each of the other curves

        :return: Pearson correlation coefficient for each of the curves.
        :rtype: dict
        """
        combination = list(combinations([self.solar, self.wind, self.load], r=2))
        combination_names = list(combinations(["solar", "wind", "load"], r=2))

        correlations = []
        for name, comb in zip(combination_names, combination):
            name = "-".join(name)
            single_result = {}
            result = self._calculate_correlation(comb[0], comb[1])
            single_result.update({"metric": "correlation", "series_type": name, "value": result})

            correlations.append(single_result)

        return correlations

    def _calculate_correlation(self, time_series_1, time_series_2):
        corr = stats.pearsonr(time_series_1.capacity_factor, time_series_2.capacity_factor)[0]
        return corr
