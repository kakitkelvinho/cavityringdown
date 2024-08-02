import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import log

plt.style.use('seaborn-v0_8-whitegrid')

@dataclass
class Ringdown:
    timetrace: np.ndarray = field(repr=False)
    t: np.ndarray = field(repr=False)
    pa_angle: int = field(default=0, init=False)
    t0: float = field(default=0., init=False, repr=False) # start of window
    window: float = field(default=1.5e-6, init=False, repr=False) # size of window

    def __post_init__(self):
        # ensure that input is numpy array
        pass

    def fit(self, t0=None, window=None, p0=[0.2, 1e-6, 0.], plot=True,):
        '''Given t0 and a window, 
        1. crops time trace
        2. takes the log
        3. fits to it
        returns: popt, pcov -- the fit parameters and covariance matrix
        '''
        if t0 == None:
            t0 = self.t0
        if window == None:
            window = self.window

        t_crop, logtimetrace = self.create_logtimetrace(t0, window)

        # fit timetrace
        popt, pcov = curve_fit(self.fit_func, xdata=t_crop, ydata=logtimetrace, p0=p0)
        
        # get residuals
        residuals = logtimetrace - self.fit_func(t_crop, *popt)
        # calculate rmse
        rmse = np.sqrt(np.sum(residuals*residuals)/(len(logtimetrace)-3))
        print(f"rmse: {rmse}")

        ### EDIT LATER



        if plot:
            self.plot_fit(logtimetrace, t_crop, popt, pcov)

        return popt, pcov

    def fit_func(self, t, a, tau, c):
        return np.log(a*np.exp(-t/tau) + c)

    def create_logtimetrace(self, t0, window, offset=1e-12):
        mask = np.logical_and(self.t >= t0, self.t <= t0+window)
        timetrace = self.timetrace[mask] - np.min(self.timetrace) + offset
        normalized_timetrace = timetrace / np.max(timetrace)
        logtimetrace = np.log(normalized_timetrace)
        t_crop = self.t[mask]
        t_crop -= t_crop[0] # to start at t=0
        return t_crop, logtimetrace



    def propagate_uncertainties(self, noise=0., t0=None, window=None):
        if t0 == None:
            t0 = self.t0
        if window == None:
            window = self.window

        min_val = ufloat(np.min(self.timetrace), noise)
        timetrace_u = [ufloat(value, noise) - min_val for value in self.timetrace]
        timetrace_u = np.array(timetrace_u)
        timetrace_u = timetrace_u[np.logical_and(self.t >= t0, self.t <= t0+window)]
        logtimetrace_u = [log(value/max(timetrace_u)) for value in timetrace_u]
        propagated_error = np.array([value.s for value in logtimetrace_u])
        var = np.mean(propagated_error*2)
        return np.sqrt(var)
        
        

    def plot_fit(self, logtimetrace, t_crop, popt, pcov):
        delta_tau = np.sqrt(pcov[1][1])
        plt.figure(figsize=(15,8))
        plt.subplot(121)
        plt.plot(self.t, self.timetrace)
        plt.title("Timetrace")
        plt.ylabel("Intensity (V)")
        plt.xlabel("Time (s)")
        plt.subplot(122)
        plt.plot(t_crop, logtimetrace, '.', ls='', color='red', label="log timetrace")
        plt.plot(t_crop, self.fit_func(t_crop, *popt), color='black', label=f"Fit: $\\tau =${popt[1]:.2e}({delta_tau:.1e})s")
        plt.xlabel("Time (s)")
        plt.ylabel("Log")
        plt.legend()
        plt.title("Log time trace")
        plt.show()


def main():
    t = np.arange(0, 2e-6, 2.5e-10)
    a, tau, c = [0.4, 1.8e-6, 0.03]
    trace = a*np.exp(-t/tau) - c
    noise_sd = a/5
    trace += np.random.normal(0, noise_sd, len(trace))

    ringdown = Ringdown(timetrace=trace, t=t)
    print(f"injected noise: {noise_sd}")

    popt, pcov = ringdown.fit()
    delta_a, delta_tau, delta_c = np.sqrt(np.diag(pcov))




if __name__ == '__main__':
    main()



        

