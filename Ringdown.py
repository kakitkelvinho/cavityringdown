import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Ringdown:
    """Class that gives a single ringdown 
    and some methods to work on it"""
    name: str # name of the ringdown i.e. ringdown47
    timetrace: list # actual trace; given in Voltages measured in Scope's channel
    tInc: float # time increment; spacing between each data point in time trace

    def __post_init__(self):
        self.t = np.arange(0, len(self.timetrace)*self.tInc, self.tInc)
        self.timetrace -= 1.001 * np.min(self.timetrace) # such that 0 is the bottom
        self.auto_window()

    def set_window(self, t0: float, t1: float):
        self.t0 = t0
        self.t1 = t1

    def auto_window(self, window_length: float = 6e-6, rolloff: float = 1e-6):
        max_time = self.t[np.argmax(self.timetrace)]
        self.t0 = max_time + rolloff
        self.t1 = self.t0 + window_length


    def crop_mask(self) -> np.ndarray:
        """Generate a mask that can window any array. Is a helper function""" 
        mask = np.logical_and(self.t >= self.t0, self.t <= self.t1)
        return mask


    def crop_timetrace(self) -> np.ndarray:
        """Crop timetrace."""
        mask = self.crop_mask()
        return self.timetrace[mask]

    def crop_time(self) -> np.ndarray:
        """Returns the windowed
        section of the full time array"""
        mask = self.crop_mask()
        return self.t[mask]

    def log_timetrace(self) -> np.ndarray:
        """Crop a time trace between t0 and t1.
        Offset it so the log does not throw an error.
        Then take a log of timetrace."""
        return np.log(self.crop_timetrace())

    def fit_timetrace(self):
        """Crop, log and then fit timetrace
        between t0 and t1.
        Return the fit object."""
        log_decay = self.log_timetrace()
        t = self.crop_time()
        t -= t[0] # so the time series starts at t=0
        fit = np.polynomial.Polynomial.fit(t, log_decay, deg=1)
        return fit


@dataclass(order=True)
class RingdownCollection:
    name: str # name of this collection i.e. 202440706_Ringdown
    path: str # abs path to the folder containing individual ringdowns
    fits: list = field(default_factory=list, compare=False, hash=False, repr=False)

    def __post_init__(self, channel='CH1(V)'):
        # load the files i.e. ringdowns
        os.path.expanduser(self.path)
        filenames = [f for f in os.listdir(self.path) if f.endswith('.csv')]
        get_tInc = lambda df: float(df.head(0).to_string().split(',')[-2].split('=')[-1].split('s')[0])
        self.ringdowns = []
        for filename in filenames:
            #print(filename)
            try:
                df = pd.read_csv(os.path.join(self.path, filename), 
                    usecols=[channel],
                    dtype=float,
                    )
                header = pd.read_csv(os.path.join(self.path, filename), nrows=0)
                tInc = get_tInc(header)
                timetrace = np.array(df[channel]).astype(float)
            except ValueError:
                continue
            ringdown = Ringdown(name=filename, timetrace=timetrace, tInc=tInc)
            self.ringdowns.append(ringdown)

    def __getitem__(self, index):
        return self.ringdowns[index]

    def fit_ringdowns(self, window=None):
        """Populate self.fits with fits to the ringdown.
        Peforms a polynomial fit f(x) = A + B x.
        Given that the mathematical model is I(t) = -1/tau t + A.
        Also returns an array of tau"""
        if window is not None:
            assert len(window) == 2
            [ringdown.set_window(t0, t1) for ringdown in self.ringdowns]
        self.fits = [ringdown.fit_timetrace() for ringdown in self.ringdowns]
        return [-1/fit.convert().coef[1] for fit in self.fits]


        # helper function for conversion
    def convert_float(self, x):
        try: 
            # common replacements
            x = x.replace('\\x05', 'E')
            x = x.replace(' ','')
            x = x.replace('&','.')
            x = x.replace(' ','')
            x = x.replace(')','E')
            x = x.replace('D','E')
            # if E is there but the next sign is gone
            if x[-3] == 'E':
                x = f"{x[:-2]}-{x[-2:]}"
            # if E is missing
            if x[-4] != 'E':
                x = f"{x[:-3]}E{x[-3:]}" 
            if 'E' in x:
                parts = x.split('E')
                if len(parts) == 2 and not (parts[1].startswith('+') or parts[1].startswith('-')):
                    x += '-02'
            return float(x)
        except (ValueError, TypeError):
            print(f"error encountered with {x}")  
            return np.nan



def main():
    angles = np.arange(0, 190, 10)
    for angle in angles:
        print(f"Beginning parsing {angle}")
        ringdowns = RingdownCollection("test", f"/home/kelvin/LabInnsbruck/WindowsData/20240715_Ringdown/PA_{angle}")
        taus = ringdowns.fit_ringdowns()
        print(len(taus))
        print(np.mean(taus))
        print(np.argwhere(np.isnan(taus)))

if __name__ == '__main__':
    main()
    

    




