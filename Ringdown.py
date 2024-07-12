import numpy as np
import matplotlib.pyplot as plt

class RingdownSeries:
    def __init__(self, series):
        self.raw_series = series

    def __iter__(self, log=False):
        if log==True:
            return [np.log(series) for series in self.raw_series)]
        else:
            return [series for series in self.raw_series]

    def __len__(self):
        return len(self.raw_series)

    def __getitem__(self, index, log=False):
        if log:
            return np.log(self.raw_series[index])
        else:
            return self.raw_series[index]

    def __repr__(self) -> str:
        return "Ringdown series"

    def __str__(self) -> str:
        return "A collection of ringdowns."
