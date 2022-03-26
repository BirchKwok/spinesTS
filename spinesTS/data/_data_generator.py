from typing import *
import numpy as np
import copy
from spinesTS.base._const import DataTS


class DataGenerator:
    """Generate timeseries-like data

    Returns
    -------
    None
    """

    @staticmethod
    def trigonometry_ds(size: int = 1000, sin_cos_noise_fact: Iterable[float] = (0.5, 0.3, 0.2),
                        random_state: Optional[int] = None):
        """Generates a weighted combination sequence of sine and cosine waves

        Parameters
        ----------
        size: int, data points number to generate
        sin_cos_noise_fact: Tuple of float, the weights of sine, cosine and standard gaussian distributed noise
        random_state: None or int, random seed

        Returns
        -------
        spinesTS.base.DataTS
        """
        assert size is not None and isinstance(size, (int, float)) is True
        assert random_state is None or isinstance(random_state, int) is True
        assert isinstance(sin_cos_noise_fact, tuple) is True and len(sin_cos_noise_fact) == 3 and \
               np.sum(sin_cos_noise_fact) == 1
        np.random.seed(random_state)
        s = int(np.ceil(size))

        assert s > 0

        ds = np.zeros(s)
        ds = ds + np.array([np.sin(i) for i in range(s)]) * sin_cos_noise_fact[0] + \
             np.array([np.cos(i) for i in range(s)]) * sin_cos_noise_fact[1] + \
             np.array([np.random.randn() for i in range(s)]) * sin_cos_noise_fact[2]

        return DataTS(ds)

    @staticmethod
    def white_noise(size=1000, mean=0., std=1., random_state=None):
        """Generates a sequence of white noise(gaussian distribution noise)

        Parameters
        ----------
        size: int, data points number to generate
        mean: float, the mean of gaussian distribution noise, default to 0.
        std: float, the standard deviation (std) of gaussian distribution noise, default to 1.
        random_state: None or int, random seed

        Returns
        -------
        spinesTS.base.DataTS
        """
        assert size is not None and isinstance(size, (int, float)) is True
        assert random_state is None or isinstance(random_state, int) is True
        assert isinstance(mean, float) is True and isinstance(std, float) is True

        np.random.seed(random_state)
        s = int(np.ceil(size))

        assert s > 0

        ds = np.random.normal(mean, std, size=s)

        return DataTS(ds)

    @staticmethod
    def random_walk(size=1000, started_zero=True, random_state=None):
        """Generates a sequence of random walk data point

        Parameters
        ----------
        size: int, data points number to generate
        started_zero: bool, whether to start from zero
        random_state: None or int, random seed

        Returns
        -------
        spinesTS.base.DataTS
        """
        assert size is not None and isinstance(size, (int, float))
        assert random_state is None or isinstance(random_state, int)
        assert isinstance(started_zero, bool) is True

        np.random.seed(random_state)
        s = int(np.ceil(size))

        if started_zero is True:
            start_p = 0.
        else:
            start_p = np.random.randn()

        res = [start_p]

        while len(res) < s:
            _ = copy.deepcopy(res[-1])
            mid = copy.deepcopy(_)
            _ += np.random.randn()
            if _ != mid:
                res.append(_)

        ds = np.array(res)

        return DataTS(ds)
