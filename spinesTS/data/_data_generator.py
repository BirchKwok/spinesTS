from typing import *
import copy

import numpy as np


class DataGenerator:
    """Generate timeseries-like data
    """

    @staticmethod
    def trigonometry_ds(size: int = 1000, sin_cos_noise_fact: Sized = (0.5, 0.3, 0.2),
                        random_state: Optional[int] = None):
        """Generates a weighted combination sequence of sine and cosine waves

        Parameters
        ----------
        size : int, data points number to generate
        sin_cos_noise_fact : Tuple of float, the weights of sine, cosine and standard gaussian distributed noise
        random_state : None or int, random seed

        Returns
        -------
        numpy.ndarray
        """
        assert size is not None and isinstance(size, (int, float)) is True
        assert random_state is None or isinstance(random_state, int) is True
        assert len(sin_cos_noise_fact) == 3
        np.random.seed(random_state)
        s = int(np.ceil(size))

        assert s > 0

        ds = np.zeros(s)
        ds = ds + np.array([np.sin(i) for i in range(s)]) * sin_cos_noise_fact[0] + \
             np.array([np.cos(i) for i in range(s)]) * sin_cos_noise_fact[1] + \
             np.array([np.random.randn() for i in range(s)]) * sin_cos_noise_fact[2]

        return ds

    @staticmethod
    def white_noise(size=1000, mean=0., std=1., random_state=None):
        """Generates a sequence of white noise(gaussian distribution noise)

        Parameters
        ----------
        size : int, data points number to generate
        mean : float, the mean of gaussian distribution noise, default to 0.
        std : float, the standard deviation (std) of gaussian distribution noise, default to 1.
        random_state : None or int, random seed

        Returns
        -------
        numpy.ndarray
        """
        assert size is not None and isinstance(size, (int, float)) is True
        assert random_state is None or isinstance(random_state, int) is True
        assert isinstance(mean, float) is True and isinstance(std, float) is True

        np.random.seed(random_state)
        s = int(np.ceil(size))

        assert s > 0

        ds = np.random.normal(mean, std, size=s)

        return ds

    @staticmethod
    def random_walk(size=1000, started_zero=True, random_state=None):
        """Generates a sequence of random walk data point

        Parameters
        ----------
        size : int, data points number to generate
        started_zero : bool, whether to start from zero
        random_state : None or int, random seed

        Returns
        -------
        numpy.ndarray
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

        return ds


class RandomEventGenerator:
    """Generate a random time series with true seasonal and temporal trends,
        intermingled with Gaussian noise

    Parameters
    ----------
    size : int, number of time collection points.
    seasons : int, seasonal cycle length.
    random_state : Random seed.

    Returns
    -------
    None
    """
    def __init__(self, size=10000, seasons=None, sin_cos_noise_fact=(0.5, 0.3, 0.2), stacking_level=6, random_state=None):
        assert isinstance(stacking_level, int)
        self.seasons = seasons
        self.size = size
        self.random_state = random_state
        self.sin_cos_noise_fact = sin_cos_noise_fact
        self.stacking_level = stacking_level

    def _seasonal(self):
        """Get seasonal trends data.

        Returns
        -------
        numpy.ndarray
        """
        np.random.seed(self.random_state)
        s = np.random.randn(self.seasons)
        return np.concatenate(
            (np.repeat(s, self.size // self.seasons),
             s[:self.size % self.seasons] if self.size % self.seasons != 0 else []))

    def _noise(self):
        """Get Gaussian noise.

        Returns
        -------
        numpy.ndarray
        """
        np.random.seed(self.random_state)
        return np.random.randn(self.size)

    def _trend(self, sin_cos_noise_fact):
        """Get temporal trends.

        Parameters
        ----------
        sin_cos_noise_fact : Tuple of float, the weights of sine, cosine and standard gaussian distributed noise

        Returns
        -------
        numpy.ndarray
        """
        np.random.seed(self.random_state)
        return DataGenerator().trigonometry_ds(size=self.size, random_state=self.random_state,
                                               sin_cos_noise_fact=sin_cos_noise_fact)

    def event(self):
        """Get event data
        results = 10 * seasonal + 8 * trend + noise

        Returns
        -------
        numpy.ndarray
        """
        sin_cos_noise_facts = [np.random.choice(self.sin_cos_noise_fact, 3) for i in range(self.stacking_level)]
        if self.seasons is not None:
            x = self._seasonal() * 10 + self._trend(sin_cos_noise_facts[0]) * 8 + self._noise()

            if self.stacking_level > 1:
                for i in sin_cos_noise_facts[1:]:
                    x += self._trend(i) * 8
        else:
            x = self._trend(sin_cos_noise_facts[0]) * 8 + self._noise()

            if self.stacking_level > 1:
                for i in sin_cos_noise_facts[1:]:
                    x += self._trend(i) * 8
        return x
